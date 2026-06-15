## Group Specific Component

### CuTile
Our ideas were:
- trying to reproduce the paper https://www.mdpi.com/2079-9292/15/5/1034 (speedups of kernel fusing)
- trying to implement + benchmark some ideas of this https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf (e.g. more tiling for reuse on different hardware levels)

### XDNA
- Reproduce the light-field tensor-ring decomposition from assignment 6 on XDNA
- 'acspx,bspy->abcyx'