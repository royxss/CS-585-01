#Process Json File "C:\Users\SROY\Documents\CodeBase\PythonWorkspace\CheckoutCart.json"

import json
#from pprint import pprint  -- using pprint will order pairs

with open(r"C:\Users\SROY\Documents\CodeBase\PythonWorkspace\CheckoutCart.json") as f:
    json_data = json.load(f)
    #print(json_data) #Prints the whole Json data
    #print(json_data.get('ShippingInstructions')) #Print Shipping Instructions
    print(json_data['ShippingInstructions']['Phone']) #Print Phone under Shipping Instruction
    
    print("++++++++++++++++++++")
    for i in json_data:
        print(i['Phone'])
