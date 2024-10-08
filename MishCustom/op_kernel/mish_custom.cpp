#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // 定义缓冲区的数量为2

// 定义自定义的 KernelMish 类，用于实现 Mish 运算的自定义内核
class KernelMish {
public:
    // 内核类的构造函数，使用 `__aicore__` 关键词表示这是在 AI Core 上执行的代码
    __aicore__ inline KernelMish() {}

    /**
    * @brief Init 函数负责初始化全局内存、局部缓存以及块和Tile的长度。
    *
    * @param x 输入数据的全局内存地址
    * @param y 输出数据的全局内存地址
    * @param totalLength 输入数据的总长度
    * @param tileNum 每个块内的数据将被进一步划分为多少个Tile
    */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // 确保块的数量不为0，否则输出错误信息
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        // 计算每个块需要处理的数据长度
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;

        // 确保tile的数量不为0，否则输出错误信息
        ASSERT(tileNum != 0 && "tile num can not be zero!");

        // 计算每个Tile的数据长度
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 初始化全局内存中输入和输出数据的缓存区域
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * GetBlockIdx(),
            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * GetBlockIdx(),
            this->blockLength);

        // 初始化队列和缓冲区，用于存储计算中间数据
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(copyBuffer, this->tileLength * sizeof(DTYPE_X));

    }

    /**
    * @brief Process 函数负责执行主循环，包括数据拷贝和计算。
    */
    __aicore__ inline void Process()
    {
        // 计算循环次数，等于 tileNum 乘以缓冲区数量
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            // 依次进行输入数据拷贝、计算以及输出数据拷贝
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    /**
    * @brief CopyIn 函数从全局内存将数据拷贝到局部内存
    *
    * @param progress 当前处理进度
    */
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从输入队列中分配一个局部张量，用于存储输入数据
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();

        // 将全局内存中的数据拷贝到局部张量中
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);

        // 将局部张量加入到输入队列中
        inQueueX.EnQue(xLocal);
    }

    /**
    * @brief Compute 函数执行具体的Mish计算操作
    *
    * @param progress 当前处理进度
    */
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列中获取一个局部张量
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();

        // 为输出分配一个局部张量
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // 为中间计算结果分配临时张量
        LocalTensor<DTYPE_X> tmpTensor = tmpBuffer.Get<DTYPE_X>();
        LocalTensor<DTYPE_X> xCopy = copyBuffer.Get<DTYPE_X>();

        // 定义计算过程中的常量
        DTYPE_X oneAdd = 1;
        
        /**************Mish算子公式**************
        Mish(x) = x*tanh(Softplus(x))
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        Softplus(x) = ln(1 + exp(x))
        ***************************************/

        // 复制x的值
        Copy(xCopy, xLocal, this->tileLength);

        // 计算 Softplus(x) = ln(1 + exp(x))
        Exp(xLocal, xLocal, this->tileLength);
        Adds(xLocal, xLocal, oneAdd, this->tileLength);
        Ln(xLocal, xLocal, this->tileLength);

        // 计算 tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        Exp(xLocal, xLocal, this->tileLength);
        Reciprocal(yLocal, xLocal, this->tileLength);
        Sub(tempLocal, xLocal, yLocal, this->tileLength);
        Add(yLocal, xLocal, yLocal, this->tileLength);
        Div(tempLocal, tempLocal, yLocal, this->tileLength);

        // 计算 Mish(x) = x*tanh(Softplus(x))
        Mul(yLocal, xCopy, tempLocal, this->tileLength);        
	    outQueueY.EnQue<DTYPE_Y>(yLocal);

        // 将输出张量放入输出队列
        outQueueY.EnQue<DTYPE_Y>(yLocal);

        // 释放局部张量
        inQueueX.FreeTensor(xLocal);
    }

    /**
    * @brief CopyOut 函数将局部内存中的结果拷贝回全局内存
    *
    * @param progress 当前处理进度
    */
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列中获取一个局部张量
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        // 将局部内存中的结果拷贝到全局内存
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);

        // 释放局部张量
        outQueueY.FreeTensor(yLocal);
    }

private:
    // 定义用于存储数据的管道和队列
    TPipe pipe;

    // 输入和输出队列，深度为缓冲区数量
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // 全局张量，用于存储全局内存中的输入和输出
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    // 定义临时缓冲区，用于中间计算
    TBuf<QuePosition::VECCALC> tmpBuffer;
    TBuf<QuePosition::VECCALC> copyBuffer;

    // 存储块的长度、Tile数量和Tile长度
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

/**
* @brief 自定义的内核函数，通过 Init 初始化操作，并调用 Process 执行计算
*
* @param x 输入数据的全局内存地址
* @param y 输出数据的全局内存地址
* @param workspace 工作空间的地址
* @param tiling 分块信息的地址
*/
extern "C" __global__ __aicore__ void mish_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    // 获取分块数据
    GET_TILING_DATA(tiling_data, tiling);

    // 创建 KernelMish 对象
    KernelMish op;

    // 调用 Init 和 Process 函数，进行初始化和计算
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
