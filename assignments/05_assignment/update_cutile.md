# update cutile for more than 5 dims

in project root

`git clone https://github.com/NVIDIA/cutile-python.git`

update:
```
diff --git a/cext/tile_kernel.cpp b/cext/tile_kernel.cpp
index d13ba78..4f2c9e2 100644
--- a/cext/tile_kernel.cpp
+++ b/cext/tile_kernel.cpp
@@ -68,7 +68,7 @@ constexpr uint8_t BYTE_BITWIDTH = 8;

 constexpr uint8_t DIVISOR_16 = 16;

-constexpr uint8_t TMA_MAX_NDIM = 5;
+constexpr uint8_t TMA_MAX_NDIM = 10;
```

`python3 get-pip.py`
`python -m pip install -e cutile-python/`