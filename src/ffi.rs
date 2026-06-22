use std::ffi::{c_int, c_uint, c_void};

pub type Pcre2Size = usize;
pub type Pcre2Code8 = c_void;
pub type Pcre2CompileContext8 = c_void;
pub type Pcre2MatchData8 = c_void;
pub type Pcre2GeneralContext8 = c_void;
pub type Pcre2MatchContext8 = c_void;

pub const PCRE2_ERROR_NOMATCH: c_int = -1;
pub const PCRE2_JIT_COMPLETE: c_uint = 0x00000001;
pub const PCRE2_UCP: c_uint = 0x00020000;
pub const PCRE2_UTF: c_uint = 0x00080000;

#[link(name = "pcre2-8")]
unsafe extern "C" {
    pub fn pcre2_compile_8(
        pattern: *const u8,
        length: Pcre2Size,
        options: c_uint,
        error_code: *mut c_int,
        error_offset: *mut Pcre2Size,
        compile_context: *mut Pcre2CompileContext8,
    ) -> *mut Pcre2Code8;

    pub fn pcre2_code_free_8(code: *mut Pcre2Code8);

    pub fn pcre2_jit_compile_8(code: *mut Pcre2Code8, options: c_uint) -> c_int;

    pub fn pcre2_match_data_create_from_pattern_8(
        code: *const Pcre2Code8,
        general_context: *mut Pcre2GeneralContext8,
    ) -> *mut Pcre2MatchData8;

    pub fn pcre2_match_data_free_8(match_data: *mut Pcre2MatchData8);

    pub fn pcre2_match_8(
        code: *const Pcre2Code8,
        subject: *const u8,
        length: Pcre2Size,
        start_offset: Pcre2Size,
        options: c_uint,
        match_data: *mut Pcre2MatchData8,
        match_context: *mut Pcre2MatchContext8,
    ) -> c_int;

    pub fn pcre2_jit_match_8(
        code: *const Pcre2Code8,
        subject: *const u8,
        length: Pcre2Size,
        start_offset: Pcre2Size,
        options: c_uint,
        match_data: *mut Pcre2MatchData8,
        match_context: *mut Pcre2MatchContext8,
    ) -> c_int;

    pub fn pcre2_get_ovector_pointer_8(match_data: *mut Pcre2MatchData8) -> *mut Pcre2Size;

    pub fn pcre2_get_error_message_8(
        error_code: c_int,
        buffer: *mut u8,
        buffer_length: Pcre2Size,
    ) -> c_int;
}
