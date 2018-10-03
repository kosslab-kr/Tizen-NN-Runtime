# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class MeanOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsMeanOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MeanOptions()
        x.Init(buf, n + offset)
        return x

    # MeanOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # MeanOptions
    def KeepDims(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
        return 0


def MeanOptionsStart(builder):
    builder.StartObject(1)


def MeanOptionsAddKeepDims(builder, keepDims):
    builder.PrependBoolSlot(0, keepDims, 0)


def MeanOptionsEnd(builder):
    return builder.EndObject()