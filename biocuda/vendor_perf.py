"""Vendor-published peak FP16 tensor-core TFLOPS (dense, no sparsity)."""

VENDOR_FP16_TC_TFLOPS = {
    "T4":         65.0,    # Turing, 65 TFLOPS FP16 TC
    "V100":      125.0,    # Volta SXM2,  ~125 TFLOPS FP16 TC
    "A100":      312.0,    # A100 80GB,   312 TFLOPS FP16 TC dense
    "A10":       125.0,    # Ampere GA102, 125 TFLOPS FP16 TC dense
    "L4":        121.0,    # Ada AD104,   121 TFLOPS FP16 TC dense
    "L40":       181.0,    # Ada AD102,   181 TFLOPS FP16 TC dense
    "RTX3090":   142.0,    # Ampere GA102, 142 TFLOPS FP16 TC dense
    "RTX4090":   330.3,    # Ada AD102,   330 TFLOPS FP16 TC dense
    "H100_SXM5": 989.0,    # Hopper SXM5, 989 TFLOPS FP16 TC dense
    "H100_PCIE": 756.0,    # Hopper PCIe, 756 TFLOPS FP16 TC dense
}
