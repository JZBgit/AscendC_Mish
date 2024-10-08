
#include "register/tilingdata_base.h"
/**
这里定义了tiling数据结构的字段totalLength和tileNum，它们分别表示输入数据的总长度和分块数目。
通过REGISTER_TILING_DATA_CLASS将MishCustomTilingData与算子MishCustom进行绑定。
**/
namespace optiling {
	BEGIN_TILING_DATA_DEF(MishCustomTilingData)
	// 定义tiling结构体成员变量
	TILING_DATA_FIELD_DEF(uint32_t, totalLength);
	TILING_DATA_FIELD_DEF(uint32_t, tileNum);
	END_TILING_DATA_DEF;
	REGISTER_TILING_DATA_CLASS(MishCustom, MishCustomTilingData)
}
