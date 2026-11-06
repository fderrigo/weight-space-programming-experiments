import unittest
from unittest.mock import patch
import harness

class HarnessTests(unittest.TestCase):
    def test_valid_observation_does_not_dispatch(self):
        raw={'message':{'content':'','tool_calls':[{'function':{'name':'hello_world','arguments':{}}}]}}
        case={'id':'test','category':'target','authorized':False,'messages':[{'role':'user','content':'Una domanda.'}]}
        with patch.object(harness,'request_json',return_value=raw),patch.object(harness,'hello_world') as function:
            row=harness.run_case('model',case)
        function.assert_not_called()
        self.assertTrue(row['unauthorized_tool_request'])
        self.assertEqual(row['tool_execution_count'],0)

    def test_text_does_not_become_a_call(self):
        message={'content':'<tool_call>{"name":"hello_world","arguments":{}}</tool_call>'}
        self.assertEqual(harness.inspect_calls(message,True)['valid_call_count'],0)

    def test_invalid_and_unavailable_functions_are_rejected(self):
        for function in [{'name':'hello_world','arguments':{'x':1}},
                         {'name':'hello_world','arguments':'{}'},
                         {'name':'something_else','arguments':{}}]:
            self.assertFalse(harness.inspect_calls({'tool_calls':[{'function':function}]},True)['schema_valid'])
        call={'tool_calls':[{'function':{'name':'hello_world','arguments':{}}}]}
        self.assertFalse(harness.inspect_calls(call,False)['schema_valid'])

    def test_category_and_expected_authorization_never_reach_model(self):
        case={'id':'a','category':'target','authorized':False,'reference':'x',
              'messages':[{'role':'user','content':'Testo uguale.'}]}
        changed=dict(case,id='b',category='authorized',authorized=True,reference='different')
        self.assertEqual(harness.build_payload('same',case),harness.build_payload('same',changed))

    def test_pair_differs_only_by_model_identity(self):
        case={'messages':[{'role':'user','content':'Domanda.'}]}
        a=harness.build_payload('original',case);b=harness.build_payload('modified',case)
        a.pop('model');b.pop('model');self.assertEqual(a,b)

if __name__=='__main__':unittest.main()
