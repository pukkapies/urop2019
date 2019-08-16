# see https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module

from _ctypes import PyObj_FromPtr
import json
import re

class NoIndent(object): # value wrapper
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # save copy of any keyword argument values needed for use here
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC
        json_repr = super(MyEncoder, self).encode(obj) # default JSON repr

        # replace any marked-up object ids in the JSON repr with the value returned from the json.dumps() of the corresponding wrapped object
        for match in self.regex.finditer(json_repr):
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # replace the matched id string with json formatted representation of the corresponding object
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)
        return json_repr