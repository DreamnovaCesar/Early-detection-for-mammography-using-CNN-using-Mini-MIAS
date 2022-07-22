# example of loading the mnist dataset

nested_dict = { 'dictA': {'key_1': 'value_1'},
                'dictB': {'key_2': 'value_2'}}

print(nested_dict['dictA'])
print(nested_dict['dictA']['key_1'])

for i, value in enumerate(nested_dict):
    for k, value1 in enumerate(nested_dict):
        print(str(value) + ' ---- ' + str(value1))

