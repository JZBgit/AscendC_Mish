#!/bin/bash
echo "[Ascend310B1] Generating MishCustom_00b2b0b8ab8f50db439d6cb44263785b ..."
opc $1 --main_func=mish_custom --input_param=/root/MishCustom/MishCustom/build_out/op_kernel/binary/ascend310b/gen/MishCustom_00b2b0b8ab8f50db439d6cb44263785b_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/MishCustom_00b2b0b8ab8f50db439d6cb44263785b.json ; then
  echo "$2/MishCustom_00b2b0b8ab8f50db439d6cb44263785b.json not generated!"
  exit 1
fi

if ! test -f $2/MishCustom_00b2b0b8ab8f50db439d6cb44263785b.o ; then
  echo "$2/MishCustom_00b2b0b8ab8f50db439d6cb44263785b.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating MishCustom_00b2b0b8ab8f50db439d6cb44263785b Done"
