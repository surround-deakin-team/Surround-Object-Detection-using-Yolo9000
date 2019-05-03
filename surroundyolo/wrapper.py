import os
import json
from surround import Surround, Wrapper, AllowedTypes
from stages import ValidateData, YoloData

class PipelineWrapper(Wrapper):
    def __init__(self):
        surround = Surround([ValidateData()], __name__)
        super().__init__(surround)
        print('Wrapper Init')

    def run(self, input_data):
        print('Wrapper Run')
        print(input_data)
        data = YoloData(input_data)
        self.surround.process(data)
        print(data.output_data)
        return {data.output_data}
