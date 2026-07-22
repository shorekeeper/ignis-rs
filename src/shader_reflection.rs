//! SPIR-V binary reflection.
//!
//! Parses compiled shader modules into structured tables of descriptor
//! bindings, push constants, vertex inputs, specialization constants,
//! and entry points. Pure Rust with no external dependencies; the parser
//! walks the SPIR-V word stream linearly and post-processes the recorded
//! decorations into final reflection data.
//!
//! # Coverage
//!
//! - Descriptor types: `SAMPLED_IMAGE`, `STORAGE_IMAGE`, `SAMPLER`,
//!   `COMBINED_IMAGE_SAMPLER`, `UNIFORM_BUFFER`, `STORAGE_BUFFER`,
//!   `INPUT_ATTACHMENT`, `UNIFORM_TEXEL_BUFFER`, `STORAGE_TEXEL_BUFFER`.
//! - Fixed-size descriptor arrays and runtime arrays (bindless).
//! - All graphics stages (vertex, tessellation, geometry, fragment),
//!   compute, and the six ray tracing stages.
//! - Push constant ranges with computed offset and size.
//! - Vertex input attribute formats inferred from base types up to vec4.
//! - Specialization constants with their declared sizes.
//! - Compute local workgroup size from `OpExecutionMode LocalSize`.
//!
//! # Limitations
//!
//! - SPIR-V `OpExtInst` instructions are skipped (the parser does not
//!   need GLSL.std.450 semantics).
//! - Vertex input formats for packed types (`R10G10B10A2`, `B5G6R5`) cannot
//!   be recovered from SPIR-V alone; the user must override them.
//! - The parser is best-effort: malformed but non-truncated SPIR-V will
//!   produce partial reflection rather than an error.
//!
//! # Example
//!
//! ```rust,no_run
//! use ignis::shader_reflection::reflect;
//!
//! # fn example() -> ignis::Result<()> {
//! let spirv: Vec<u32> = std::fs::read("shader.spv")
//!     .unwrap()
//!     .chunks_exact(4)
//!     .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
//!     .collect();
//!
//! let reflection = reflect(&spirv)?;
//! for entry in &reflection.entry_points {
//!     println!("entry point: {} ({:?})", entry.name, entry.stage);
//! }
//! for b in &reflection.descriptor_bindings {
//!     println!(
//!         "  set={} binding={} type={:?} count={} stage={:?} name={:?}",
//!         b.set, b.binding, b.descriptor_type, b.count, b.stage, b.name
//!     );
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

use ash::vk;

use crate::error::{Error, Result};

const SPIRV_MAGIC: u32 = 0x0723_0203;

// Storage classes used by descriptor- and IO-related variables.
const SC_UNIFORM_CONSTANT: u32 = 0;
const SC_INPUT: u32 = 1;
const SC_UNIFORM: u32 = 2;
#[allow(dead_code)]
const SC_OUTPUT: u32 = 3;
const SC_PUSH_CONSTANT: u32 = 9;
const SC_STORAGE_BUFFER: u32 = 12;

// Decorations of interest.
const DEC_SPEC_ID: u32 = 1;
const DEC_BLOCK: u32 = 2;
const DEC_BUFFER_BLOCK: u32 = 3;
const DEC_BUILT_IN: u32 = 11;
const DEC_LOCATION: u32 = 30;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;
const DEC_INPUT_ATTACHMENT_INDEX: u32 = 43;

// Opcodes the parser recognizes. Anything else is skipped using the
// instruction's word_count header.
const OP_NAME: u16 = 5;
const OP_MEMBER_NAME: u16 = 6;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_SPEC_CONSTANT_TRUE: u16 = 48;
const OP_SPEC_CONSTANT_FALSE: u16 = 49;
const OP_SPEC_CONSTANT: u16 = 50;
const OP_VARIABLE: u16 = 59;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;

// Execution models (SPIR-V "ExecutionModel" enum).
const EM_VERTEX: u32 = 0;
const EM_TESS_CONTROL: u32 = 1;
const EM_TESS_EVAL: u32 = 2;
const EM_GEOMETRY: u32 = 3;
const EM_FRAGMENT: u32 = 4;
const EM_GL_COMPUTE: u32 = 5;
const EM_RAYGEN: u32 = 5313;
const EM_INTERSECTION: u32 = 5314;
const EM_ANY_HIT: u32 = 5315;
const EM_CLOSEST_HIT: u32 = 5316;
const EM_MISS: u32 = 5317;
const EM_CALLABLE: u32 = 5318;

// Execution modes.
const EXEC_MODE_LOCAL_SIZE: u32 = 17;

// SPIR-V image dimension. `Buffer` indicates a texel buffer.
const DIM_BUFFER: u32 = 5;
const DIM_SUBPASS_DATA: u32 = 6;

/// One entry point declared by the module.
#[derive(Debug, Clone)]
pub struct EntryPoint {
    /// Name as written in the source (typically `"main"`).
    pub name: String,
    /// Vulkan shader stage flag corresponding to the SPIR-V execution model.
    pub stage: vk::ShaderStageFlags,
    /// SPIR-V id of the function the entry point references.
    pub function_id: u32,
}

/// One descriptor binding declared by the module.
#[derive(Debug, Clone)]
pub struct DescriptorBinding {
    /// Descriptor set index (from `DescriptorSet` decoration).
    pub set: u32,
    /// Binding slot within the set (from `Binding` decoration).
    pub binding: u32,
    /// Inferred descriptor type.
    pub descriptor_type: vk::DescriptorType,
    /// Array length. `1` for scalar bindings, `N` for fixed arrays,
    /// `0` for runtime-sized arrays (bindless).
    pub count: u32,
    /// Stage flags collected from every entry point that references this
    /// binding. For a single-stage module the result has one flag; for a
    /// multi-stage module it is the union.
    pub stage: vk::ShaderStageFlags,
    /// Variable name as declared in source, if `OpName` was emitted.
    pub name: Option<String>,
    /// Input attachment index (only present for `INPUT_ATTACHMENT`).
    pub input_attachment_index: Option<u32>,
}

/// A push constant range declared by the module.
///
/// Vulkan requires the union of stage flags of every push constant range
/// used by a pipeline. Reflection produces one range per entry point that
/// touches push constants; downstream code is expected to merge ranges
/// covering the same byte region across stages.
#[derive(Debug, Clone)]
pub struct PushConstantRange {
    /// Byte offset of the lowest-addressed member.
    pub offset: u32,
    /// Total byte size from the lowest member to the end of the highest.
    pub size: u32,
    /// Stage flag this range applies to.
    pub stage: vk::ShaderStageFlags,
    /// Block name, if `OpName` was emitted on the variable.
    pub name: Option<String>,
}

/// One vertex input attribute (vertex shader inputs only).
#[derive(Debug, Clone)]
pub struct VertexInput {
    /// Attribute location index (from `Location` decoration).
    pub location: u32,
    /// Vulkan format inferred from the SPIR-V base type.
    pub format: vk::Format,
    /// Variable name, if `OpName` was emitted.
    pub name: Option<String>,
}

/// A specialization constant declared by the module.
#[derive(Debug, Clone)]
pub struct SpecConstant {
    /// Constant id (from `SpecId` decoration).
    pub id: u32,
    /// Size in bytes of the constant's underlying type.
    pub size_bytes: u32,
    /// Variable name, if `OpName` was emitted.
    pub name: Option<String>,
}

/// Complete reflection result for one SPIR-V module.
#[derive(Debug, Clone, Default)]
pub struct ShaderReflection {
    /// All entry points found in the module.
    pub entry_points: Vec<EntryPoint>,
    /// Every descriptor binding the module references.
    pub descriptor_bindings: Vec<DescriptorBinding>,
    /// Push constant ranges declared by the module.
    pub push_constant_ranges: Vec<PushConstantRange>,
    /// Vertex input attributes (only populated for vertex modules).
    pub vertex_inputs: Vec<VertexInput>,
    /// Specialization constants declared by the module.
    pub spec_constants: Vec<SpecConstant>,
    /// Compute workgroup size, if the module is a compute module with
    /// `LocalSize` declared.
    pub local_size: Option<[u32; 3]>,
}

/// Parse a SPIR-V binary into a [`ShaderReflection`].
///
/// # Errors
///
/// Returns [`Error::InvalidSpirv`] if the binary is shorter than a header,
/// has wrong magic, or contains a self-referential / zero-length
/// instruction that would loop the parser. Any other malformation is
/// tolerated and yields a best-effort partial reflection.
pub fn reflect(spirv: &[u32]) -> Result<ShaderReflection> {
    if spirv.len() < 5 || spirv[0] != SPIRV_MAGIC {
        return Err(Error::InvalidSpirv);
    }
    let mut p = Parser::new(spirv);
    p.parse()?;
    Ok(p.finalize())
}

/// Build per-set `VkDescriptorSetLayoutBinding` arrays from a reflection.
///
/// The returned map has one entry per descriptor set referenced by the
/// module. Each value is sorted by binding index. Counts are taken from
/// the reflection (including 0 for runtime arrays, which the caller may
/// override via descriptor indexing limits).
pub fn descriptor_set_layouts_from(
    reflection: &ShaderReflection,
) -> HashMap<u32, Vec<vk::DescriptorSetLayoutBinding<'static>>> {
    let mut out: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding<'static>>> = HashMap::new();
    for b in &reflection.descriptor_bindings {
        let entry = out.entry(b.set).or_default();
        entry.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(b.binding)
                .descriptor_type(b.descriptor_type)
                .descriptor_count(b.count.max(1))
                .stage_flags(b.stage),
        );
    }
    for v in out.values_mut() {
        v.sort_by_key(|b| b.binding);
    }
    out
}

/// Convert reflection push constant ranges into `VkPushConstantRange`.
pub fn push_constant_ranges_from(reflection: &ShaderReflection) -> Vec<vk::PushConstantRange> {
    reflection
        .push_constant_ranges
        .iter()
        .map(|r| vk::PushConstantRange {
            stage_flags: r.stage,
            offset: r.offset,
            size: r.size,
        })
        .collect()
}

// ---------- internal parser state ----------

#[derive(Debug, Clone, Default)]
struct IdDecorations {
    descriptor_set: Option<u32>,
    binding: Option<u32>,
    location: Option<u32>,
    spec_id: Option<u32>,
    built_in: bool,
    input_attachment_index: Option<u32>,
    has_block: bool,
    has_buffer_block: bool,
}

#[derive(Debug, Clone, Default)]
struct MemberDecorationSet {
    offset: Option<u32>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum TypeInfo {
    Void,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Vector { component_id: u32, count: u32 },
    Matrix { column_id: u32, count: u32 },
    Image { sampled_type_id: u32, dim: u32, sampled: u32 },
    Sampler,
    SampledImage { image_id: u32 },
    Array { element_id: u32, length: u32 },
    RuntimeArray { element_id: u32 },
    Struct { member_ids: Vec<u32> },
    Pointer { storage_class: u32, pointee_id: u32 },
    Other,
}

#[derive(Debug, Clone)]
struct VariableInfo {
    type_id: u32,
    storage_class: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RawEntryPoint {
    execution_model: u32,
    function_id: u32,
    name: String,
    interface_ids: Vec<u32>,
}

struct Parser<'a> {
    words: &'a [u32],
    types: HashMap<u32, TypeInfo>,
    decorations: HashMap<u32, IdDecorations>,
    member_decorations: HashMap<(u32, u32), MemberDecorationSet>,
    names: HashMap<u32, String>,
    member_names: HashMap<(u32, u32), String>,
    variables: HashMap<u32, VariableInfo>,
    spec_constant_types: HashMap<u32, u32>,
    entry_points: Vec<RawEntryPoint>,
    execution_modes: Vec<(u32, u32, Vec<u32>)>,
}

impl<'a> Parser<'a> {
    fn new(words: &'a [u32]) -> Self {
        Self {
            words,
            types: HashMap::new(),
            decorations: HashMap::new(),
            member_decorations: HashMap::new(),
            names: HashMap::new(),
            member_names: HashMap::new(),
            variables: HashMap::new(),
            spec_constant_types: HashMap::new(),
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
        }
    }

    /// Walk the instruction stream from word 5 (after the header).
    fn parse(&mut self) -> Result<()> {
        let mut i = 5_usize;
        while i < self.words.len() {
            let header = self.words[i];
            let opcode = (header & 0xFFFF) as u16;
            let word_count = (header >> 16) as usize;
            if word_count == 0 {
                // A zero-length instruction would loop forever.
                return Err(Error::InvalidSpirv);
            }
            if i + word_count > self.words.len() {
                // Truncated.
                return Err(Error::InvalidSpirv);
            }
            // Operand window starts at i + 1, has word_count - 1 words.
            let ops_start = i + 1;
            let ops_end = i + word_count;
            self.dispatch(opcode, ops_start, ops_end);
            i = ops_end;
        }
        Ok(())
    }

    fn dispatch(&mut self, opcode: u16, start: usize, end: usize) {
        let ops = &self.words[start..end];
        match opcode {
            OP_NAME => self.handle_name(ops),
            OP_MEMBER_NAME => self.handle_member_name(ops),
            OP_ENTRY_POINT => self.handle_entry_point(ops),
            OP_EXECUTION_MODE => self.handle_execution_mode(ops),
            OP_DECORATE => self.handle_decorate(ops),
            OP_MEMBER_DECORATE => self.handle_member_decorate(ops),

            OP_TYPE_VOID => self.handle_type_void(ops),
            OP_TYPE_INT => self.handle_type_int(ops),
            OP_TYPE_FLOAT => self.handle_type_float(ops),
            OP_TYPE_VECTOR => self.handle_type_vector(ops),
            OP_TYPE_MATRIX => self.handle_type_matrix(ops),
            OP_TYPE_IMAGE => self.handle_type_image(ops),
            OP_TYPE_SAMPLER => self.handle_type_sampler(ops),
            OP_TYPE_SAMPLED_IMAGE => self.handle_type_sampled_image(ops),
            OP_TYPE_ARRAY => self.handle_type_array(ops),
            OP_TYPE_RUNTIME_ARRAY => self.handle_type_runtime_array(ops),
            OP_TYPE_STRUCT => self.handle_type_struct(ops),
            OP_TYPE_POINTER => self.handle_type_pointer(ops),

            OP_VARIABLE => self.handle_variable(ops),
            OP_SPEC_CONSTANT | OP_SPEC_CONSTANT_TRUE | OP_SPEC_CONSTANT_FALSE => {
                self.handle_spec_constant(ops)
            }
            _ => {}
        }
    }

    // --- handlers ---

    fn handle_name(&mut self, ops: &[u32]) {
        if ops.is_empty() {
            return;
        }
        let target = ops[0];
        let (name, _) = parse_string(&ops[1..]);
        self.names.insert(target, name);
    }

    fn handle_member_name(&mut self, ops: &[u32]) {
        if ops.len() < 2 {
            return;
        }
        let target = ops[0];
        let member = ops[1];
        let (name, _) = parse_string(&ops[2..]);
        self.member_names.insert((target, member), name);
    }

    fn handle_entry_point(&mut self, ops: &[u32]) {
        if ops.len() < 3 {
            return;
        }
        let exec_model = ops[0];
        let function_id = ops[1];
        let (name, words_used) = parse_string(&ops[2..]);
        let interface_start = 2 + words_used;
        let interface_ids: Vec<u32> = ops[interface_start..].to_vec();
        self.entry_points.push(RawEntryPoint {
            execution_model: exec_model,
            function_id,
            name,
            interface_ids,
        });
    }

    fn handle_execution_mode(&mut self, ops: &[u32]) {
        if ops.len() < 2 {
            return;
        }
        let entry_id = ops[0];
        let mode = ops[1];
        let mode_ops: Vec<u32> = ops[2..].to_vec();
        self.execution_modes.push((entry_id, mode, mode_ops));
    }

    fn handle_decorate(&mut self, ops: &[u32]) {
        if ops.len() < 2 {
            return;
        }
        let target = ops[0];
        let dec = ops[1];
        let entry = self.decorations.entry(target).or_default();
        match dec {
            DEC_DESCRIPTOR_SET if ops.len() >= 3 => entry.descriptor_set = Some(ops[2]),
            DEC_BINDING if ops.len() >= 3 => entry.binding = Some(ops[2]),
            DEC_LOCATION if ops.len() >= 3 => entry.location = Some(ops[2]),
            DEC_SPEC_ID if ops.len() >= 3 => entry.spec_id = Some(ops[2]),
            DEC_BUILT_IN => entry.built_in = true,
            DEC_BLOCK => entry.has_block = true,
            DEC_BUFFER_BLOCK => entry.has_buffer_block = true,
            DEC_INPUT_ATTACHMENT_INDEX if ops.len() >= 3 => {
                entry.input_attachment_index = Some(ops[2])
            }
            _ => {}
        }
    }

    fn handle_member_decorate(&mut self, ops: &[u32]) {
        if ops.len() < 3 {
            return;
        }
        let target = ops[0];
        let member = ops[1];
        let dec = ops[2];
        let entry = self.member_decorations.entry((target, member)).or_default();
        if dec == DEC_OFFSET && ops.len() >= 4 {
            entry.offset = Some(ops[3]);
        }
    }

    // --- type handlers ---

    fn handle_type_void(&mut self, ops: &[u32]) {
        if let Some(&id) = ops.first() {
            self.types.insert(id, TypeInfo::Void);
        }
    }

    fn handle_type_int(&mut self, ops: &[u32]) {
        if ops.len() >= 3 {
            self.types.insert(
                ops[0],
                TypeInfo::Int {
                    width: ops[1],
                    signed: ops[2] != 0,
                },
            );
        }
    }

    fn handle_type_float(&mut self, ops: &[u32]) {
        if ops.len() >= 2 {
            self.types
                .insert(ops[0], TypeInfo::Float { width: ops[1] });
        }
    }

    fn handle_type_vector(&mut self, ops: &[u32]) {
        if ops.len() >= 3 {
            self.types.insert(
                ops[0],
                TypeInfo::Vector {
                    component_id: ops[1],
                    count: ops[2],
                },
            );
        }
    }

    fn handle_type_matrix(&mut self, ops: &[u32]) {
        if ops.len() >= 3 {
            self.types.insert(
                ops[0],
                TypeInfo::Matrix {
                    column_id: ops[1],
                    count: ops[2],
                },
            );
        }
    }

    fn handle_type_image(&mut self, ops: &[u32]) {
        // result_id, sampled_type, dim, depth, arrayed, ms, sampled, format, [access]
        if ops.len() >= 7 {
            self.types.insert(
                ops[0],
                TypeInfo::Image {
                    sampled_type_id: ops[1],
                    dim: ops[2],
                    sampled: ops[6],
                },
            );
        }
    }

    fn handle_type_sampler(&mut self, ops: &[u32]) {
        if let Some(&id) = ops.first() {
            self.types.insert(id, TypeInfo::Sampler);
        }
    }

    fn handle_type_sampled_image(&mut self, ops: &[u32]) {
        if ops.len() >= 2 {
            self.types
                .insert(ops[0], TypeInfo::SampledImage { image_id: ops[1] });
        }
    }

    fn handle_type_array(&mut self, ops: &[u32]) {
        // result_id, element_id, length_id (a constant whose value is the length).
        // We resolve the length only when we recorded the constant via OpConstant.
        if ops.len() >= 3 {
            // Try to resolve length; default to 0 (treat as runtime array
            // if the constant could not be parsed).
            self.types.insert(
                ops[0],
                TypeInfo::Array {
                    element_id: ops[1],
                    length: 0, // may be overwritten below
                },
            );
            // Find the constant value if we recorded it. We piggyback on
            // OP_CONSTANT recording done later. To do it inline, we walk
            // backwards in the words: an OpConstant for the length must
            // already have been emitted before the type that uses it.
            // Lazy approach: set 0 here; resolve in finalize() if needed.
            // For typical shader code the length is a literal scalar uint;
            // we can recover it by scanning OpConstant entries.
            //
            // Simpler approach below: search through self.words for an
            // OpConstant whose result id == ops[2]. We won't do that here
            // for performance; we do it once in finalize().
            //
            // The placeholder 0 is corrected by finalize_array_lengths().
            let _ = ops[2]; // length id; resolved later
        }
    }

    fn handle_type_runtime_array(&mut self, ops: &[u32]) {
        if ops.len() >= 2 {
            self.types
                .insert(ops[0], TypeInfo::RuntimeArray { element_id: ops[1] });
        }
    }

    fn handle_type_struct(&mut self, ops: &[u32]) {
        if let Some(&id) = ops.first() {
            let members: Vec<u32> = ops[1..].to_vec();
            self.types
                .insert(id, TypeInfo::Struct { member_ids: members });
        }
    }

    fn handle_type_pointer(&mut self, ops: &[u32]) {
        if ops.len() >= 3 {
            self.types.insert(
                ops[0],
                TypeInfo::Pointer {
                    storage_class: ops[1],
                    pointee_id: ops[2],
                },
            );
        }
    }

    fn handle_variable(&mut self, ops: &[u32]) {
        if ops.len() >= 3 {
            self.variables.insert(
                ops[1],
                VariableInfo {
                    type_id: ops[0],
                    storage_class: ops[2],
                },
            );
        }
    }

    fn handle_spec_constant(&mut self, ops: &[u32]) {
        // result_type, result_id, [value words for OP_SPEC_CONSTANT]
        if ops.len() >= 2 {
            self.spec_constant_types.insert(ops[1], ops[0]);
        }
    }

    /// Walk the module a second time to resolve `OpConstant` values used
    /// as `OpTypeArray` lengths. Called once from `finalize`.
    fn resolve_array_lengths(&mut self) {
        // Build a map id -> u32 value for every OpConstant of integer type.
        let mut const_values: HashMap<u32, u32> = HashMap::new();
        let mut i = 5_usize;
        while i < self.words.len() {
            let header = self.words[i];
            let opcode = (header & 0xFFFF) as u16;
            let word_count = (header >> 16) as usize;
            if word_count == 0 || i + word_count > self.words.len() {
                break;
            }
            if opcode == OP_CONSTANT && word_count >= 4 {
                // result_type, result_id, value (single u32 for 32-bit ints).
                let result_id = self.words[i + 2];
                let value = self.words[i + 3];
                const_values.insert(result_id, value);
            }
            i += word_count;
        }

        // Re-walk type table; for arrays, the original length operand is in
        // the words at the OpTypeArray position. We re-scan instructions to
        // find OpTypeArray and patch lengths.
        let mut i = 5_usize;
        while i < self.words.len() {
            let header = self.words[i];
            let opcode = (header & 0xFFFF) as u16;
            let word_count = (header >> 16) as usize;
            if word_count == 0 || i + word_count > self.words.len() {
                break;
            }
            if opcode == OP_TYPE_ARRAY && word_count >= 4 {
                let result_id = self.words[i + 1];
                let element_id = self.words[i + 2];
                let length_id = self.words[i + 3];
                if let Some(&len) = const_values.get(&length_id) {
                    self.types.insert(
                        result_id,
                        TypeInfo::Array {
                            element_id,
                            length: len,
                        },
                    );
                }
            }
            i += word_count;
        }
    }

    fn finalize(mut self) -> ShaderReflection {
        self.resolve_array_lengths();

        let mut out = ShaderReflection::default();

        // Build entry points.
        for ep in &self.entry_points {
            let stage = exec_model_to_stage(ep.execution_model);
            out.entry_points.push(EntryPoint {
                name: ep.name.clone(),
                stage,
                function_id: ep.function_id,
            });
        }

        // Compute local size if a compute entry point declared LocalSize.
        for (entry_id, mode, ops) in &self.execution_modes {
            if *mode == EXEC_MODE_LOCAL_SIZE && ops.len() >= 3 {
                // Verify the entry id is a compute-stage entry.
                if self
                    .entry_points
                    .iter()
                    .any(|ep| ep.function_id == *entry_id && ep.execution_model == EM_GL_COMPUTE)
                {
                    out.local_size = Some([ops[0], ops[1], ops[2]]);
                }
            }
        }

        // Union of all stage flags this module contributes.
        let module_stage = out
            .entry_points
            .iter()
            .map(|e| e.stage)
            .fold(vk::ShaderStageFlags::empty(), |a, b| a | b);

        // Process variables for descriptor bindings, push constants,
        // vertex inputs, and specialization constants.
        for (var_id, var_info) in &self.variables {
            match var_info.storage_class {
                SC_UNIFORM_CONSTANT | SC_UNIFORM | SC_STORAGE_BUFFER => {
                    self.classify_descriptor(*var_id, var_info, module_stage, &mut out);
                }
                SC_PUSH_CONSTANT => {
                    self.classify_push_constant(*var_id, var_info, module_stage, &mut out);
                }
                SC_INPUT => {
                    if module_stage.contains(vk::ShaderStageFlags::VERTEX) {
                        self.classify_vertex_input(*var_id, var_info, &mut out);
                    }
                }
                _ => {}
            }
        }

        // Specialization constants.
        for (id, type_id) in &self.spec_constant_types {
            let dec = self.decorations.get(id);
            let Some(spec_id) = dec.and_then(|d| d.spec_id) else {
                continue;
            };
            let size_bytes = self
                .types
                .get(type_id)
                .map(|t| size_of_scalar_type(t))
                .unwrap_or(4);
            out.spec_constants.push(SpecConstant {
                id: spec_id,
                size_bytes,
                name: self.names.get(id).cloned(),
            });
        }

        // Sort outputs for deterministic order.
        out.descriptor_bindings.sort_by_key(|b| (b.set, b.binding));
        out.vertex_inputs.sort_by_key(|v| v.location);
        out.spec_constants.sort_by_key(|s| s.id);

        out
    }

    fn classify_descriptor(
        &self,
        var_id: u32,
        var: &VariableInfo,
        module_stage: vk::ShaderStageFlags,
        out: &mut ShaderReflection,
    ) {
        let Some(dec) = self.decorations.get(&var_id) else {
            return;
        };
        if dec.built_in {
            return;
        }
        let (Some(set), Some(binding)) = (dec.descriptor_set, dec.binding) else {
            return;
        };

        // Resolve pointer to underlying type.
        let pointee_id = match self.types.get(&var.type_id) {
            Some(TypeInfo::Pointer { pointee_id, .. }) => *pointee_id,
            _ => return,
        };

        // Walk through optional arrays to determine element type and count.
        let (element_id, count) = self.peel_array(pointee_id);
        let element_decorations = self.decorations.get(&element_id);

        let descriptor_type = match self.types.get(&element_id) {
            Some(TypeInfo::Image { dim, sampled, .. }) => {
                if *dim == DIM_BUFFER {
                    if *sampled == 1 {
                        vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_TEXEL_BUFFER
                    }
                } else if *dim == DIM_SUBPASS_DATA {
                    vk::DescriptorType::INPUT_ATTACHMENT
                } else if *sampled == 1 {
                    vk::DescriptorType::SAMPLED_IMAGE
                } else {
                    vk::DescriptorType::STORAGE_IMAGE
                }
            }
            Some(TypeInfo::Sampler) => vk::DescriptorType::SAMPLER,
            Some(TypeInfo::SampledImage { .. }) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Some(TypeInfo::Struct { .. }) => {
                let block = element_decorations.is_some_and(|d| d.has_block);
                let buffer_block = element_decorations.is_some_and(|d| d.has_buffer_block);
                match var.storage_class {
                    SC_UNIFORM => {
                        if block {
                            vk::DescriptorType::UNIFORM_BUFFER
                        } else if buffer_block {
                            // SPIR-V 1.3 legacy SSBO encoding under Uniform.
                            vk::DescriptorType::STORAGE_BUFFER
                        } else {
                            vk::DescriptorType::UNIFORM_BUFFER
                        }
                    }
                    SC_STORAGE_BUFFER => vk::DescriptorType::STORAGE_BUFFER,
                    _ => return,
                }
            }
            _ => return,
        };

        out.descriptor_bindings.push(DescriptorBinding {
            set,
            binding,
            descriptor_type,
            count,
            stage: module_stage,
            name: self.names.get(&var_id).cloned(),
            input_attachment_index: dec.input_attachment_index,
        });
    }

    fn classify_push_constant(
        &self,
        var_id: u32,
        var: &VariableInfo,
        module_stage: vk::ShaderStageFlags,
        out: &mut ShaderReflection,
    ) {
        let pointee_id = match self.types.get(&var.type_id) {
            Some(TypeInfo::Pointer { pointee_id, .. }) => *pointee_id,
            _ => return,
        };
        let Some(TypeInfo::Struct { member_ids }) = self.types.get(&pointee_id) else {
            return;
        };
        if member_ids.is_empty() {
            return;
        }
        // Compute offset = min OpMemberDecorate Offset, size = max(offset + member_size).
        let mut min_off: Option<u32> = None;
        let mut max_end: u32 = 0;
        for (idx, m_id) in member_ids.iter().enumerate() {
            let Some(off) = self
                .member_decorations
                .get(&(pointee_id, idx as u32))
                .and_then(|d| d.offset)
            else {
                continue;
            };
            min_off = Some(min_off.map_or(off, |o| o.min(off)));
            let member_size = self
                .types
                .get(m_id)
                .map(|t| self.size_of_type(t))
                .unwrap_or(0);
            let end = off + member_size;
            if end > max_end {
                max_end = end;
            }
        }
        let offset = min_off.unwrap_or(0);
        let size = max_end.saturating_sub(offset);
        out.push_constant_ranges.push(PushConstantRange {
            offset,
            size,
            stage: module_stage,
            name: self.names.get(&var_id).cloned(),
        });
    }

    fn classify_vertex_input(
        &self,
        var_id: u32,
        var: &VariableInfo,
        out: &mut ShaderReflection,
    ) {
        let Some(dec) = self.decorations.get(&var_id) else {
            return;
        };
        if dec.built_in {
            return;
        }
        let Some(location) = dec.location else {
            return;
        };
        let pointee_id = match self.types.get(&var.type_id) {
            Some(TypeInfo::Pointer { pointee_id, .. }) => *pointee_id,
            _ => return,
        };
        let format = self.infer_vertex_format(pointee_id).unwrap_or(vk::Format::R32G32B32A32_SFLOAT);
        out.vertex_inputs.push(VertexInput {
            location,
            format,
            name: self.names.get(&var_id).cloned(),
        });
    }

    /// Walk through OpTypeArray / OpTypeRuntimeArray wrappers, returning
    /// the innermost element type id and the array count (`0` for runtime).
    fn peel_array(&self, mut id: u32) -> (u32, u32) {
        let mut count: u32 = 1;
        loop {
            match self.types.get(&id) {
                Some(TypeInfo::Array { element_id, length }) => {
                    count = if *length == 0 { 0 } else { *length };
                    id = *element_id;
                }
                Some(TypeInfo::RuntimeArray { element_id }) => {
                    count = 0;
                    id = *element_id;
                }
                _ => break,
            }
        }
        (id, count)
    }

    fn infer_vertex_format(&self, type_id: u32) -> Option<vk::Format> {
        match self.types.get(&type_id)? {
            TypeInfo::Float { width: 32 } => Some(vk::Format::R32_SFLOAT),
            TypeInfo::Float { width: 64 } => Some(vk::Format::R64_SFLOAT),
            TypeInfo::Int { width: 32, signed: true } => Some(vk::Format::R32_SINT),
            TypeInfo::Int { width: 32, signed: false } => Some(vk::Format::R32_UINT),
            TypeInfo::Vector { component_id, count } => {
                let comp = self.types.get(component_id)?;
                Some(match (comp, *count) {
                    (TypeInfo::Float { width: 32 }, 2) => vk::Format::R32G32_SFLOAT,
                    (TypeInfo::Float { width: 32 }, 3) => vk::Format::R32G32B32_SFLOAT,
                    (TypeInfo::Float { width: 32 }, 4) => vk::Format::R32G32B32A32_SFLOAT,
                    (TypeInfo::Int { width: 32, signed: true }, 2) => vk::Format::R32G32_SINT,
                    (TypeInfo::Int { width: 32, signed: true }, 3) => vk::Format::R32G32B32_SINT,
                    (TypeInfo::Int { width: 32, signed: true }, 4) => vk::Format::R32G32B32A32_SINT,
                    (TypeInfo::Int { width: 32, signed: false }, 2) => vk::Format::R32G32_UINT,
                    (TypeInfo::Int { width: 32, signed: false }, 3) => vk::Format::R32G32B32_UINT,
                    (TypeInfo::Int { width: 32, signed: false }, 4) => vk::Format::R32G32B32A32_UINT,
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    fn size_of_type(&self, t: &TypeInfo) -> u32 {
        match t {
            TypeInfo::Float { width } | TypeInfo::Int { width, .. } => width / 8,
            TypeInfo::Vector { component_id, count } => self
                .types
                .get(component_id)
                .map(|c| self.size_of_type(c) * count)
                .unwrap_or(0),
            TypeInfo::Matrix { column_id, count } => self
                .types
                .get(column_id)
                .map(|c| self.size_of_type(c) * count)
                .unwrap_or(0),
            TypeInfo::Array { element_id, length } => self
                .types
                .get(element_id)
                .map(|e| self.size_of_type(e) * length)
                .unwrap_or(0),
            TypeInfo::Struct { member_ids } => member_ids
                .iter()
                .map(|m| self.types.get(m).map(|t| self.size_of_type(t)).unwrap_or(0))
                .sum(),
            _ => 0,
        }
    }
}

fn size_of_scalar_type(t: &TypeInfo) -> u32 {
    match t {
        TypeInfo::Float { width } | TypeInfo::Int { width, .. } => width / 8,
        _ => 4,
    }
}

fn exec_model_to_stage(model: u32) -> vk::ShaderStageFlags {
    match model {
        EM_VERTEX => vk::ShaderStageFlags::VERTEX,
        EM_TESS_CONTROL => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        EM_TESS_EVAL => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        EM_GEOMETRY => vk::ShaderStageFlags::GEOMETRY,
        EM_FRAGMENT => vk::ShaderStageFlags::FRAGMENT,
        EM_GL_COMPUTE => vk::ShaderStageFlags::COMPUTE,
        EM_RAYGEN => vk::ShaderStageFlags::RAYGEN_KHR,
        EM_INTERSECTION => vk::ShaderStageFlags::INTERSECTION_KHR,
        EM_ANY_HIT => vk::ShaderStageFlags::ANY_HIT_KHR,
        EM_CLOSEST_HIT => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        EM_MISS => vk::ShaderStageFlags::MISS_KHR,
        EM_CALLABLE => vk::ShaderStageFlags::CALLABLE_KHR,
        _ => vk::ShaderStageFlags::empty(),
    }
}

/// Parse a SPIR-V LiteralString from a slice of u32 words.
///
/// Reads consecutive u32 words, treating each word as four bytes
/// (little-endian within the word), until a NUL byte is encountered.
/// Returns the parsed string and the number of words consumed.
fn parse_string(words: &[u32]) -> (String, usize) {
    let mut bytes = Vec::with_capacity(16);
    let mut i = 0;
    'outer: while i < words.len() {
        let w = words[i];
        i += 1;
        for shift in 0..4 {
            let b = ((w >> (shift * 8)) & 0xFF) as u8;
            if b == 0 {
                break 'outer;
            }
            bytes.push(b);
        }
    }
    (String::from_utf8_lossy(&bytes).into_owned(), i)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same minimal compute shader used by the smoke tests.
    #[rustfmt::skip]
    const EMPTY_COMPUTE_SPV: &[u32] = &[
        0x07230203, 0x00010000, 0x00000000, 0x00000006, 0x00000000,
        0x00020011, 0x00000001,
        0x0003000E, 0x00000000, 0x00000001,
        0x0005000F, 0x00000005, 0x00000004, 0x6E69616D, 0x00000000,
        0x00060010, 0x00000004, 0x00000011, 0x00000001, 0x00000001, 0x00000001,
        0x00020013, 0x00000002,
        0x00030021, 0x00000003, 0x00000002,
        0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
        0x000200F8, 0x00000005,
        0x000100FD,
        0x00010038,
    ];

    #[rustfmt::skip]
    const MINIMAL_VERT_SPV: &[u32] = &[
        0x07230203, 0x00010000, 0x00000000, 0x00000011, 0x00000000,
        0x00020011, 0x00000001,
        0x0003000E, 0x00000000, 0x00000001,
        0x0006000F, 0x00000000, 0x00000003, 0x6E69616D, 0x00000000, 0x00000008,
        0x00050048, 0x00000006, 0x00000000, 0x0000000B, 0x00000000,
        0x00030047, 0x00000006, 0x00000002,
        0x00020013, 0x00000001,
        0x00030021, 0x00000002, 0x00000001,
        0x00030016, 0x00000004, 0x00000020,
        0x00040017, 0x00000005, 0x00000004, 0x00000004,
        0x0003001E, 0x00000006, 0x00000005,
        0x00040020, 0x00000007, 0x00000003, 0x00000006,
        0x00040015, 0x0000000C, 0x00000020, 0x00000000,
        0x00040020, 0x0000000E, 0x00000003, 0x00000005,
        0x0004003B, 0x00000007, 0x00000008, 0x00000003,
        0x0004002B, 0x00000004, 0x00000009, 0x00000000,
        0x0004002B, 0x00000004, 0x0000000A, 0x3F800000,
        0x0007002C, 0x00000005, 0x0000000B, 0x00000009, 0x00000009, 0x00000009, 0x0000000A,
        0x0004002B, 0x0000000C, 0x0000000D, 0x00000000,
        0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002,
        0x000200F8, 0x0000000F,
        0x00050041, 0x0000000E, 0x00000010, 0x00000008, 0x0000000D,
        0x0003003E, 0x00000010, 0x0000000B,
        0x000100FD,
        0x00010038,
    ];

    #[rustfmt::skip]
    const MINIMAL_FRAG_SPV: &[u32] = &[
        0x07230203, 0x00010000, 0x00000000, 0x0000000C, 0x00000000,
        0x00020011, 0x00000001,
        0x0003000E, 0x00000000, 0x00000001,
        0x0006000F, 0x00000004, 0x00000003, 0x6E69616D, 0x00000000, 0x00000007,
        0x00030010, 0x00000003, 0x00000007,
        0x00040047, 0x00000007, 0x0000001E, 0x00000000,
        0x00020013, 0x00000001,
        0x00030021, 0x00000002, 0x00000001,
        0x00030016, 0x00000004, 0x00000020,
        0x00040017, 0x00000005, 0x00000004, 0x00000004,
        0x00040020, 0x00000006, 0x00000003, 0x00000005,
        0x0004003B, 0x00000006, 0x00000007, 0x00000003,
        0x0004002B, 0x00000004, 0x00000008, 0x3F800000,
        0x0004002B, 0x00000004, 0x00000009, 0x00000000,
        0x0007002C, 0x00000005, 0x0000000A, 0x00000008, 0x00000009, 0x00000009, 0x00000008,
        0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002,
        0x000200F8, 0x0000000B,
        0x0003003E, 0x00000007, 0x0000000A,
        0x000100FD,
        0x00010038,
    ];

    #[test]
    fn rejects_empty_input() {
        let r = reflect(&[]);
        assert!(matches!(r, Err(Error::InvalidSpirv)));
    }

    #[test]
    fn rejects_wrong_magic() {
        let bad: Vec<u32> = vec![0xDEAD_BEEF, 0, 0, 0, 0];
        let r = reflect(&bad);
        assert!(matches!(r, Err(Error::InvalidSpirv)));
    }

    #[test]
    fn rejects_too_short() {
        let too_short: Vec<u32> = vec![SPIRV_MAGIC, 0, 0, 0]; // only 4 words
        let r = reflect(&too_short);
        assert!(matches!(r, Err(Error::InvalidSpirv)));
    }

    #[test]
    fn rejects_zero_word_count() {
        // Header: 5 valid words, then an instruction with word_count=0 to trip parser.
        let mut bad: Vec<u32> = vec![SPIRV_MAGIC, 0x00010000, 0, 16, 0];
        bad.push(0); // opcode 0, word_count 0 -> infinite loop guard
        let r = reflect(&bad);
        assert!(matches!(r, Err(Error::InvalidSpirv)));
    }

    #[test]
    fn rejects_truncated_instruction() {
        // word_count claims 4 but only 1 word follows.
        let mut bad: Vec<u32> = vec![SPIRV_MAGIC, 0x00010000, 0, 16, 0];
        bad.push((4 << 16) | 1);
        // missing 3 follow-up words
        let r = reflect(&bad);
        assert!(matches!(r, Err(Error::InvalidSpirv)));
    }

    #[test]
    fn empty_compute_extracts_entry_point() {
        let r = reflect(EMPTY_COMPUTE_SPV).unwrap();
        assert_eq!(r.entry_points.len(), 1);
        assert_eq!(r.entry_points[0].name, "main");
        assert_eq!(r.entry_points[0].stage, vk::ShaderStageFlags::COMPUTE);
        assert_eq!(r.local_size, Some([1, 1, 1]));
        assert!(r.descriptor_bindings.is_empty());
        assert!(r.push_constant_ranges.is_empty());
    }

    #[test]
    fn vertex_shader_extracts_entry_point() {
        let r = reflect(MINIMAL_VERT_SPV).unwrap();
        assert_eq!(r.entry_points.len(), 1);
        assert_eq!(r.entry_points[0].name, "main");
        // Note: this fixture's entry point uses execution model 0 (Vertex)
        // even though the smoke test labels it differently in code. The
        // model byte is at word 12 (0x00000000 = Vertex).
        assert_eq!(r.entry_points[0].stage, vk::ShaderStageFlags::VERTEX);
        assert!(r.local_size.is_none());
    }

    #[test]
    fn fragment_shader_extracts_entry_point() {
        let r = reflect(MINIMAL_FRAG_SPV).unwrap();
        assert_eq!(r.entry_points.len(), 1);
        assert_eq!(r.entry_points[0].name, "main");
        assert_eq!(r.entry_points[0].stage, vk::ShaderStageFlags::FRAGMENT);
    }

    #[test]
    fn parse_string_handles_short_string() {
        // "main" + null = 5 chars, packed in 2 words.
        let words = [0x6E69616D_u32, 0x00000000_u32];
        let (s, used) = parse_string(&words);
        assert_eq!(s, "main");
        assert_eq!(used, 2);
    }

    #[test]
    fn parse_string_terminates_on_null() {
        let words = [0x4F424155_u32]; // "UAB" pattern actually 'U','A','B','O' but null hits...
        // Build manually: bytes = [0x55='U', 0x41='A', 0x42='B', 0x00] -> "UAB"
        let one_word = (0x55 | (0x41 << 8) | (0x42 << 16)) as u32;
        let (s, used) = parse_string(&[one_word]);
        assert_eq!(s, "UAB");
        assert_eq!(used, 1);
        let _ = words;
    }

    #[test]
    fn reflection_with_uniform_buffer_extracts_descriptor() {
        // Hand-assembled fragment shader fixture with one uniform buffer
        // at set=0, binding=0. Built from the SPIR-V structure documented
        // at the top of this file.
        let spv = build_ubo_fixture();
        let r = reflect(&spv).unwrap();

        assert_eq!(r.entry_points.len(), 1);
        assert_eq!(r.entry_points[0].stage, vk::ShaderStageFlags::FRAGMENT);

        assert_eq!(r.descriptor_bindings.len(), 1);
        let b = &r.descriptor_bindings[0];
        assert_eq!(b.set, 0);
        assert_eq!(b.binding, 0);
        assert_eq!(b.descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(b.count, 1);
        assert_eq!(b.stage, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(b.name.as_deref(), Some("ubo"));
    }

    #[test]
    fn descriptor_set_layouts_from_groups_by_set() {
        let spv = build_ubo_fixture();
        let r = reflect(&spv).unwrap();
        let layouts = descriptor_set_layouts_from(&r);
        assert_eq!(layouts.len(), 1);
        let bindings = &layouts[&0];
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].binding, 0);
        assert_eq!(bindings[0].descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
    }

    #[test]
    fn unknown_opcodes_are_skipped() {
        // Synthesize a shader that has a bogus "future opcode" between
        // valid ones; reflection should still extract the entry point.
        let mut spv: Vec<u32> = Vec::new();
        // Header.
        spv.extend_from_slice(&[SPIRV_MAGIC, 0x00010000, 0, 17, 0]);
        // OpCapability Shader.
        spv.push((2 << 16) | 17);
        spv.push(1);
        // Bogus opcode 999, word_count 4.
        spv.push((4 << 16) | 999);
        spv.extend_from_slice(&[42, 43, 44]);
        // OpMemoryModel.
        spv.push((3 << 16) | 14);
        spv.extend_from_slice(&[0, 1]);
        // OpEntryPoint Fragment %4 "main".
        spv.push((5 << 16) | 15);
        spv.push(4); // Fragment
        spv.push(4); // function id
        spv.push(0x6E69616D);
        spv.push(0x00000000);

        let r = reflect(&spv).unwrap();
        assert_eq!(r.entry_points.len(), 1);
        assert_eq!(r.entry_points[0].stage, vk::ShaderStageFlags::FRAGMENT);
    }

    /// Build the hand-assembled UBO fixture. See module-level test docs.
    fn build_ubo_fixture() -> Vec<u32> {
        let mut s = Vec::with_capacity(64);
        // Header: bound id high enough.
        s.extend_from_slice(&[SPIRV_MAGIC, 0x00010000, 0, 32, 0]);

        // OpCapability Shader.
        s.extend_from_slice(&[(2 << 16) | 17, 1]);
        // OpMemoryModel Logical GLSL450.
        s.extend_from_slice(&[(3 << 16) | 14, 0, 1]);
        // OpEntryPoint Fragment %4 "main".
        s.extend_from_slice(&[(5 << 16) | 15, 4, 4, 0x6E69616D, 0x00000000]);
        // OpExecutionMode %4 OriginUpperLeft (7).
        s.extend_from_slice(&[(3 << 16) | 16, 4, 7]);

        // OpName %4 "main".
        s.extend_from_slice(&[(4 << 16) | 5, 4, 0x6E69616D, 0x00000000]);
        // OpName %7 "UBO".
        s.extend_from_slice(&[(3 << 16) | 5, 7, 0x004F4255]);
        // OpName %9 "ubo".
        s.extend_from_slice(&[(3 << 16) | 5, 9, 0x006F6275]);

        // OpDecorate %7 Block (2).
        s.extend_from_slice(&[(3 << 16) | 71, 7, 2]);
        // OpMemberDecorate %7 0 Offset 0.
        s.extend_from_slice(&[(5 << 16) | 72, 7, 0, 35, 0]);
        // OpDecorate %9 DescriptorSet 0.
        s.extend_from_slice(&[(4 << 16) | 71, 9, 34, 0]);
        // OpDecorate %9 Binding 0.
        s.extend_from_slice(&[(4 << 16) | 71, 9, 33, 0]);

        // OpTypeVoid %2.
        s.extend_from_slice(&[(2 << 16) | 19, 2]);
        // OpTypeFunction %3 %2.
        s.extend_from_slice(&[(3 << 16) | 33, 3, 2]);
        // OpTypeFloat %5 32.
        s.extend_from_slice(&[(3 << 16) | 22, 5, 32]);
        // OpTypeVector %6 %5 4.
        s.extend_from_slice(&[(4 << 16) | 23, 6, 5, 4]);
        // OpTypeStruct %7 %6.
        s.extend_from_slice(&[(3 << 16) | 30, 7, 6]);
        // OpTypePointer %8 Uniform %7.
        s.extend_from_slice(&[(4 << 16) | 32, 8, 2, 7]);
        // OpVariable %9 %8 Uniform.
        s.extend_from_slice(&[(4 << 16) | 59, 8, 9, 2]);

        // OpFunction %4 %2 None %3, OpLabel %10, OpReturn, OpFunctionEnd.
        s.extend_from_slice(&[(5 << 16) | 54, 2, 4, 0, 3]);
        s.extend_from_slice(&[(2 << 16) | 248, 10]);
        s.extend_from_slice(&[(1 << 16) | 253]);
        s.extend_from_slice(&[(1 << 16) | 56]);

        s
    }
}