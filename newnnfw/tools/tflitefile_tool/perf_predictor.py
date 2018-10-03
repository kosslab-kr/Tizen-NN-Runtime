#!/usr/bin/python

from operation import Operation


class PerfPredictor(object):
    def __init__(self, add_cycle=1, mul_cycle=1, nonlinear_cycle=1):
        self.add_cycle = add_cycle
        self.mul_cycle = mul_cycle
        self.nonlinear_cycle = nonlinear_cycle

    def PredictCycles(self, operation):
        return (operation.add_instr_num * self.add_cycle +
                operation.mul_instr_num * self.mul_cycle +
                operation.nonlinear_instr_num * self.nonlinear_cycle)
