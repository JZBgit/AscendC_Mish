#include "mish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
    /**
    * @brief TilingFunc 函数负责将输入数据进行分块（Tile）处理。
    *
    * 分块处理的优势在于能够并行计算不同块中的数据，从而提升整体计算效率。
    *
    * @param context 当前的分块上下文，包含输入输出的形状信息及其他配置。
    * @return 返回图计算状态，成功则返回 ge::GRAPH_SUCCESS。
    */
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        MishCustomTilingData tiling;

        // 定义每次计算操作需要处理的块的数量
        const uint32_t BLOCK_DIM = 8;

        // 定义在每个计算块中进一步划分的子块数量
        const uint32_t TILE_NUM = 8;

        // 获取输入数据的总长度（元素数量）
        uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

        // 设置分块维度
        context->SetBlockDim(BLOCK_DIM);

        // 保存总长度和子块数量到 tiling 对象中
        tiling.set_totalLength(totalLength);
        tiling.set_tileNum(TILE_NUM);

        // 将 tiling 数据保存到 RawTilingData 缓冲区中
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
            context->GetRawTilingData()->GetCapacity());

        // 设置 RawTilingData 的实际数据大小
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        // 获取当前工作空间的指针，并初始化第一个工作空间大小为 0
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    /**
    * @brief InferShape 函数用于定义输入与输出的形状推理逻辑。
    *
    * 该函数确保输出的形状与输入的形状保持一致。
    *
    * @param context 形状推理的上下文，包含输入输出的形状信息。
    * @return 返回图计算状态，成功则返回 GRAPH_SUCCESS。
    */
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        // 获取第一个输入张量的形状
        const gert::Shape* x1_shape = context->GetInputShape(0);

        // 获取第一个输出张量的形状
        gert::Shape* y_shape = context->GetOutputShape(0);

        // 将输入形状赋值给输出形状，确保两者相同
        *y_shape = *x1_shape;

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    /**
    * @brief MishCustom 类定义了一个自定义的 Mish 算子。
    *
    * 该算子明确了输入和输出的张量格式和数据类型（DT_FLOAT16），并指定了形状推理函数和分块函数。
    * 最后，通过 OP_ADD(MishCustom) 将该算子注册到 Ascend 编译器中。
    */
    class MishCustom : public OpDef {
    public:
        /**
        * @brief 构造函数，初始化 MishCustom 算子的输入输出及相关配置。
        *
        * @param name 算子的名称。
        */
        explicit MishCustom(const char* name) : OpDef(name)
        {
            // 定义输入张量 "x" 的属性
            this->Input("x")
                .ParamType(REQUIRED)                       // 输入为必需参数
                .DataType({ ge::DT_FLOAT16 })              // 数据类型为 float16
                .Format({ ge::FORMAT_ND })                  // 数据格式为 N 维格式
                .UnknownShapeFormat({ ge::FORMAT_ND });      // 未知形状时的数据格式

            // 定义输出张量 "y" 的属性
            this->Output("y")
                .ParamType(REQUIRED)                       // 输出为必需参数
                .DataType({ ge::DT_FLOAT16 })              // 数据类型为 float16
                .Format({ ge::FORMAT_ND })                  // 数据格式为 N 维格式
                .UnknownShapeFormat({ ge::FORMAT_ND });      // 未知形状时的数据格式

            // 设置形状推理函数
            this->SetInferShape(ge::InferShape);

            // 配置 AICore 相关设置，包括分块函数和特定的硬件配置
            this->AICore()
                .SetTiling(optiling::TilingFunc);           // 设置分块函数
            this->AICore().AddConfig("ascend310b");         // 添加 Ascend 310B 的硬件配置
        }
    };

    // 将 MishCustom 算子注册到操作定义注册表中
    OP_ADD(MishCustom);
}
