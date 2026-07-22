//! Filesystem-based shader hot-reload.
//!
//! [`ShaderWatcher`] runs a background thread that periodically polls
//! registered file paths for modification (size or mtime change). When a
//! change is detected, the file is read and the registered callback is
//! invoked with the new bytes. Useful for hot-reloading SPIR-V shaders
//! during development.
//!
//! No external dependencies (no `notify` crate). Works on every platform
//! that exposes file modification times. The polling approach has a few
//! hundred millisecond reload latency, which is fine for development
//! workflow but should not be used as a real-time event source.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::shader_watcher::ShaderWatcher;
//! # use std::time::Duration;
//! let watcher = ShaderWatcher::new(Duration::from_millis(500));
//! watcher.watch("shaders/effect.spv", |bytes| {
//!     eprintln!("shader changed; new size = {} bytes", bytes.len());
//!     // Re-parse, rebuild pipeline, swap in atomically...
//! });
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime};

use crate::error::{Error, Result};

type Callback = Box<dyn Fn(&[u8]) + Send + Sync>;

#[derive(Clone)]
struct WatchEntry {
    last_mtime: Option<SystemTime>,
    last_size: u64,
    callback: Arc<Callback>,
}

struct WatcherState {
    entries: HashMap<PathBuf, WatchEntry>,
}

struct NotifyPair {
    flag: Mutex<bool>,
    cvar: Condvar,
}

impl NotifyPair {
    fn new() -> Self {
        Self {
            flag: Mutex::new(false),
            cvar: Condvar::new(),
        }
    }
    fn signal(&self) {
        *self.flag.lock().unwrap() = true;
        self.cvar.notify_one();
    }
    fn wait(&self, dur: Duration) {
        let f = self.flag.lock().unwrap();
        let (mut f, _) = self.cvar.wait_timeout(f, dur).unwrap();
        *f = false;
    }
}

/// Watches files on disk and reloads them when they change.
///
/// Spawns a single background polling thread on construction. The thread
/// terminates when the watcher is dropped.
pub struct ShaderWatcher {
    state: Arc<Mutex<WatcherState>>,
    notify: Arc<NotifyPair>,
    shutdown: Arc<AtomicBool>,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    poll_interval: Duration,
}

impl ShaderWatcher {
    /// Create a watcher with the given poll interval.
    ///
    /// Lower intervals reduce reload latency but use more CPU. 200-500ms
    /// is appropriate for most development use cases.
    pub fn new(poll_interval: Duration) -> Arc<Self> {
        let state = Arc::new(Mutex::new(WatcherState {
            entries: HashMap::new(),
        }));
        let notify = Arc::new(NotifyPair::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let t_state = Arc::clone(&state);
        let t_notify = Arc::clone(&notify);
        let t_shutdown = Arc::clone(&shutdown);
        let t_interval = poll_interval;

        let handle = std::thread::Builder::new()
            .name("ignis-shader-watcher".into())
            .spawn(move || {
                Self::watcher_loop(&t_state, &t_notify, &t_shutdown, t_interval);
            })
            .expect("failed to spawn shader watcher thread");

        Arc::new(Self {
            state,
            notify,
            shutdown,
            handle: Mutex::new(Some(handle)),
            poll_interval,
        })
    }

    /// Register a callback to be invoked when the file at `path` changes.
    ///
    /// The callback receives the new file contents as raw bytes. If the
    /// path is already watched, the new callback replaces the old one.
    /// The first call after registration does NOT fire the callback even
    /// if the file already exists; only subsequent changes do.
    pub fn watch<P, F>(&self, path: P, callback: F)
    where
        P: AsRef<Path>,
        F: Fn(&[u8]) + Send + Sync + 'static,
    {
        let path = path.as_ref().to_path_buf();
        let (last_mtime, last_size) = match std::fs::metadata(&path) {
            Ok(m) => (m.modified().ok(), m.len()),
            Err(_) => (None, 0),
        };
        let entry = WatchEntry {
            last_mtime,
            last_size,
            callback: Arc::new(Box::new(callback)),
        };
        self.state.lock().unwrap().entries.insert(path, entry);
        self.notify.signal();
    }

    /// Stop watching a previously registered path.
    pub fn unwatch(&self, path: impl AsRef<Path>) {
        let path = path.as_ref().to_path_buf();
        self.state.lock().unwrap().entries.remove(&path);
    }

    /// Snapshot of currently watched paths.
    pub fn watched_paths(&self) -> Vec<PathBuf> {
        self.state.lock().unwrap().entries.keys().cloned().collect()
    }

    /// Poll interval this watcher was constructed with.
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    fn watcher_loop(
        state: &Mutex<WatcherState>,
        notify: &NotifyPair,
        shutdown: &AtomicBool,
        interval: Duration,
    ) {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Snapshot entries so we don't hold the lock during file IO.
            let snapshot: Vec<(PathBuf, WatchEntry)> = {
                let s = state.lock().unwrap();
                s.entries
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            };

            for (path, entry) in snapshot {
                let meta = match std::fs::metadata(&path) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let size = meta.len();
                let mtime = meta.modified().ok();

                let changed = mtime != entry.last_mtime || size != entry.last_size;
                if !changed {
                    continue;
                }

                // Read file before updating state so a failed read does
                // not advance the watermark; we will retry next tick.
                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                // Update tracking state and grab a fresh callback handle
                // under the lock; release before invoking the callback.
                let cb = {
                    let mut s = state.lock().unwrap();
                    if let Some(e) = s.entries.get_mut(&path) {
                        e.last_mtime = mtime;
                        e.last_size = size;
                        Arc::clone(&e.callback)
                    } else {
                        continue;
                    }
                };
                cb(&bytes);
            }

            if !shutdown.load(Ordering::Relaxed) {
                notify.wait(interval);
            }
        }
    }
}

impl Drop for ShaderWatcher {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.notify.signal();
        if let Some(h) = self.handle.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

/// Convert a byte slice (e.g. a file's raw contents) into a `Vec<u32>`
/// suitable for [`ShaderModule::new`](crate::ShaderModule) and
/// [`reflect`](super::shader_reflection::reflect).
///
/// The byte slice must be a multiple of 4 bytes long. Endianness is
/// little-endian per the SPIR-V binary spec.
pub fn bytes_to_spirv(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return Err(Error::InvalidSpirv);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ignis_watch_test_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn bytes_to_spirv_round_trip() {
        let words: Vec<u32> = vec![0x07230203, 0xDEADBEEF, 0xCAFEBABE];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let parsed = bytes_to_spirv(&bytes).unwrap();
        assert_eq!(parsed, words);
    }

    #[test]
    fn bytes_to_spirv_rejects_non_multiple_of_four() {
        let bytes = [0u8; 7];
        assert!(matches!(bytes_to_spirv(&bytes), Err(Error::InvalidSpirv)));
    }

    #[test]
    fn watch_and_unwatch_track_paths() {
        let watcher = ShaderWatcher::new(Duration::from_millis(50));
        let p1 = temp_path("a.spv");
        let p2 = temp_path("b.spv");
        std::fs::write(&p1, b"hello").unwrap();
        std::fs::write(&p2, b"world").unwrap();

        watcher.watch(&p1, |_| {});
        watcher.watch(&p2, |_| {});
        let mut paths = watcher.watched_paths();
        paths.sort();
        let mut expected = vec![p1.clone(), p2.clone()];
        expected.sort();
        assert_eq!(paths, expected);

        watcher.unwatch(&p1);
        assert_eq!(watcher.watched_paths(), vec![p2.clone()]);

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn callback_fires_on_modification() {
        let path = temp_path("mod.spv");
        // Initial content.
        std::fs::write(&path, b"original").unwrap();

        // Sleep so the next write produces a different mtime; many
        // filesystems have second-granularity timestamps.
        std::thread::sleep(Duration::from_millis(10));

        let watcher = ShaderWatcher::new(Duration::from_millis(50));
        let counter = Arc::new(AtomicU32::new(0));
        let last_size = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let s = Arc::clone(&last_size);
        watcher.watch(&path, move |bytes| {
            c.fetch_add(1, Ordering::Relaxed);
            s.store(bytes.len() as u32, Ordering::Relaxed);
        });

        // Modify the file. Wait a moment for mtime to differ.
        std::thread::sleep(Duration::from_millis(1100));
        std::fs::write(&path, b"modified content").unwrap();

        // Give the watcher up to 2s to detect the change.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while counter.load(Ordering::Relaxed) == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }

        assert!(
            counter.load(Ordering::Relaxed) >= 1,
            "callback never fired"
        );
        assert_eq!(last_size.load(Ordering::Relaxed), 16);

        drop(watcher);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn callback_does_not_fire_on_initial_registration() {
        let path = temp_path("initial.spv");
        std::fs::write(&path, b"data").unwrap();

        let watcher = ShaderWatcher::new(Duration::from_millis(30));
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        watcher.watch(&path, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });

        // Wait several poll intervals.
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        drop(watcher);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_does_not_fire_or_panic() {
        let watcher = ShaderWatcher::new(Duration::from_millis(30));
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        watcher.watch("/path/that/definitely/does/not/exist.spv", move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn dropping_watcher_terminates_background_thread() {
        // Build, hold briefly, drop, ensure no panic.
        let watcher = ShaderWatcher::new(Duration::from_millis(20));
        std::thread::sleep(Duration::from_millis(50));
        drop(watcher);
        // Implicitly: if the thread did not terminate, the test would hang
        // on a subsequent join. We rely on Drop::join.
    }
}