from typing import List

# MAJOR_FOOD_GROUPS = ["protein", "proteins", "vegetable", "vegetables", "grain", "grains", "sugar", "sugars",
#                      "fruit", "fruits", "dairy", "fats", "oils", "sweet", "sweets", "carbohydrate", "carbohydrates"]

MAJOR_FOOD_GROUPS = ["protein", "vegetable", "grain",  "sugar",
                     "fruit", "dairy", "fats", "oil", "sweet", "carbohydrate"]

preds = """['User: What are the food groups in this image? \\nAssistant: The food groups in this 
image include grains (from the omelette and toast), protein (from the bacon and ham), and vegetables (from the peppers and onions).']"""


def find_list_elements_in_string(lst, my_string):
    # Convert the list elements to a set for faster membership testing
    list_set = set(lst)

    # Find elements in the string that are also in the set of list elements
    found_elements = [element for element in list_set if element in my_string]

    return found_elements


# example usage
# result = find_food_groups(MAJOR_FOOD_GROUPS, preds)
# print("Elements found in the string:", result)

