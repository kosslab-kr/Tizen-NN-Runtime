# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class SequenceRNNOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSequenceRNNOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SequenceRNNOptions()
        x.Init(buf, n + offset)
        return x

    # SequenceRNNOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SequenceRNNOptions
    def TimeMajor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos)
        return 0

    # SequenceRNNOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def SequenceRNNOptionsStart(builder):
    builder.StartObject(2)


def SequenceRNNOptionsAddTimeMajor(builder, timeMajor):
    builder.PrependBoolSlot(0, timeMajor, 0)


def SequenceRNNOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)


def SequenceRNNOptionsEnd(builder):
    return builder.EndObject()
