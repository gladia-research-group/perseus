//! cuBLASLt (int8 -> int32 GEMM for PCMM/CCMM), Cargo feature `cublas`. The FFI and the typed GEMM plans are
//! pc-cublas, re-exported as `lt`; this module owns the handle creation with the runtime version check
//! (int8 exactness is validated on cuBLASLt 13.x from `/usr/local/cuda-13.2/lib64`). Without the feature `Lt::new`
//! reports itself unavailable.
use crate::{Device, Error, Result};

#[cfg(feature = "cublas")]
pub use pc_cublas as lt;

/// cuBLASLt major version the int8 exactness check was validated against.
pub const LT_MAJOR: usize = 13;

/// cuBLASLt handle, created after a device guard and the version check.
pub struct Lt {
    #[cfg(feature = "cublas")]
    pub inner: pc_cublas::Lt,
    #[cfg(not(feature = "cublas"))]
    _private: (),
}

impl Lt {
    #[cfg(feature = "cublas")]
    pub fn new(dev: &Device) -> Result<Lt> {
        if dev.is_host() {
            return Ok(Lt { inner: pc_cublas::Lt::host() });
        }
        let v = pc_cublas::version();
        if v / 10000 != LT_MAJOR {
            return Err(Error::Blas(format!("cuBLASLt {} loaded; the int8 exactness gate validated major {LT_MAJOR}", pc_cublas::version_string())));
        }
        Ok(Lt { inner: pc_cublas::Lt::new().map_err(|e| Error::Blas(e.to_string()))? })
    }
    #[cfg(not(feature = "cublas"))]
    pub fn new(_dev: &Device) -> Result<Lt> {
        Err(Error::Unavailable("cuBLASLt (build pc-cuda with feature `cublas`)"))
    }
}
