from typing import List

MAJOR_FOOD_GROUPS = ["meat", "protein", "vegetable", "grain",  "sugar",
                     "fruit", "dairy", "fats", "oil", "sweet", "carbohydrate"]
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


food_groups_dict = {
    'Fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'kiwi', 'watermelon', 'pineapple', 'mango', 'pear', 'blueberry', 'raspberry', 'blackberry', 'peach', 'plum', 'cherry', 'pomegranate', 'fig', 'cantaloupe', 'dragon fruit', 'guava', 'kiwifruit', 'passion fruit', 'cranberry', 'nectarine', 'apricot', 'papaya', 'lychee', 'star fruit (carambola)', 'persimmon', 'durian', 'jackfruit', 'tamarind', 'soursop (graviola)', 'ugli fruit (unique fruit)', 'kumquat'],

    'Vegetables': ['carrot', 'broccoli', 'spinach', 'tomato', 'cucumber', 'bell pepper', 'zucchini','potato', 'sweet potato', 'kale', 'asparagus', 'celery', 'cauliflower', 'green beans', 'peas', 'radish', 'eggplant', 'cabbage', 'lettuce', 'onion', 'garlic', 'artichoke', 'leek', 'squash', 'brussels sprouts', 'collard greens', 'beets', 'turnip', 'rutabaga', 'okra', 'swiss chard', 'bok choy', 'parsnip', 'ginger', 'fennel', 'endive', 'arugula', 'mustard greens', 'jicama', 'kohlrabi', 'daikon', 'chicory', 'watercress', 'yam', 'horseradish', 'chayote', 'pumpkin', 'mushrooms'],

    'Proteins': ['chicken','eggs','meatballs','meat','steak', 'beef', 'tofu', 'fish', 'beans', 'lentils', 'pork', 'shrimp', 'turkey', 'salmon', 'quorn', 'duck', 'lamb', 'tempeh', 'crab', 'lobster', 'ham', 'sausage', 'veal', 'bison', 'venison', 'rabbit', 'elk', 'halibut', 'trout', 'tilapia', 'catfish', 'swordfish', 'barramundi', 'mussels', 'oysters', 'scallops', 'cod', 'sardines', 'anchovies', 'clams', 'octopus', 'squid (calamari)', 'tuna', 'haddock', 'mackerel', 'seitan', 'peas', 'edamame', 'chickpeas', 'black beans', 'kidney beans', 'navy beans', 'pinto beans', 'soybeans', 'almonds', 'walnuts', 'cashews', 'pistachios', 'sunflower seeds', 'pumpkin seeds', 'chia seeds', 'flaxseeds', 'hemp seeds'],

    'Grains': ['bread', 'crackers', 'whole wheat bread','rye bread', 'buns', 'rolls', 'bagels', 'pretzels', 'cornbread','rice','pasta','spaghetti', 'fettuccine', 'penne', 'rigatoni', 'farfalle', 'ravioli', 'tortellini', 'lasagna', 'orzo', 'linguine', 'cannelloni', 'gnocchi', 'tagliatelle', 'pappardelle', 'orecchiette', 'capellini', 'macaroni', 'fusilli', 'rotini', 'quinoa', 'oats', 'barley', 'wheat', 'bulgur', 'farro', 'couscous', 'millet', 'sorghum', 'buckwheat', 'amaranth', 'teff', 'spelt', 'rye', 'cornmeal', 'polenta', 'wild rice', 'fonio', 'triticale', 'einkorn', 'kamut', 'freekeh', 'sorghum', 'chickpea flour', 'semolina', 'chia seeds', 'quinoa flour', 'almond flour', 'coconut flour'],
    
    'Fast Foods': ['hamburger', 'pizza', 'french fries','fries', 'hot dog', 'chicken nuggets', 'tacos', 'burritos', 'fried chicken', 'onion rings', 'mozzarella sticks', 'cheeseburger','gyro', 'kebab', 'quesadilla', 'chicken wings', 'nachos', 'bagels', 'pretzel', 'sliders', 'corn dog', 'fish and chips', 'potato wedges', 'philly cheesesteak', 'chili dog', 'loaded fries', 'sub sandwich', 'falafel sandwich', 'pita wrap', 'churros', 'popcorn chicken', 'meatball sub', 'bacon cheeseburger', 'chicken sandwich', 'tater tots', 'jalapeño poppers', 'calzone', 'stromboli',  'egg rolls', 'spring rolls', 'empanadas', 'big mac', 'whopper', 'nuggets', 'mcflurry',  'chicken tenders', 'spicy chicken sandwich', 'curly fries', 'taco supreme', 'cinnabon'],

    'Dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cottage cheese', 'cream', 'cheddar', 'greek yogurt', 'feta', 'mozzarella', 'goat cheese', 'swiss cheese', 'brie', 'ricotta', 'sour cream', 'parmesan', 'havarti', 'provolone', 'blue cheese', 'gouda', 'asiago', 'colby jack', 'mascarpone', 'queso fresco', 'queso blanco', 'camembert', 'roquefort', 'stilton', 'emmental', 'gruyère', 'halloumi', 'manchego', 'monterey jack', 'munster', 'neufchâtel', 'paneer', 'pecorino', 'romano', 'smetana', 'tilsit', 'velveeta', 'burrata', 'clotted cream', 'crème fraîche', 'double cream', 'edam', 'fontina', 'jarlsberg', 'kefir', 'limburger', 'mascarpone', 'oaxaca', 'quark', 'reblochon', 'ricotta salata', 'stracciatella'],

    'Sweets': ['waffles', 'pancakes', 'syrup', 'frosting', 'marzipan', 'sweet rolls','chocolate', 'ice cream', 'cookies', 'cake', 'candy', 'brownies', 'pie', 'donuts', 'gelato', 'macarons', 'cupcakes', 'muffins', 'caramel', 'fudge', 'pudding', 'sorbet', 'tiramisu', 'cheesecake', 'fruit sorbet', 'marshmallows', 'cotton candy', 'toffee', 'truffles', 'shortbread', 'baklava', 'churros', 'eclairs', 'pavlova', 'biscotti', 'cannoli', 'tarte tatin', 'key lime pie', 'lemon meringue pie', 'scones', 'croissants', 'danish pastry', 'profiteroles', 'flan', 'crème brûlée', 'mousse', 'opera cake', 'panna cotta', 'rugelach', 'sacher torte', 'strudel', 'swiss roll', 'angel food cake', 'pound cake', 'carrot cake', 'red velvet cake', 'baked alaska', 'banoffee pie', 'blondies', 'rice krispie treats'],

    'Fats and Oils': ['olive oil', 'avocado', 'nuts', 'butter', 'coconut oil', 'sunflower oil', 'almond butter', 'sesame oil', 'flaxseed oil', 'pumpkin seed oil', 'walnut oil', 'peanut butter', 'grapeseed oil', 'hazelnut oil', 'cashew butter', 'pistachio oil', 'macadamia nut oil', 'palm oil', 'canola oil', 'soybean oil', 'chia seeds', 'hemp oil', 'avocado oil', 'lard', 'beef tallow', 'duck fat', 'ghee'],

    'Beverages': ['water', 'tea', 'coffee', 'orange juice', 'apple juice', 'milkshake', 'smoothie', 'soda', 'lemonade', 'iced tea', 'hot chocolate', 'wine', 'beer', 'cocktail', 'mocktail', 'fruit punch', 'coconut water', 'vegetable juice', 'kombucha', 'chai latte', 'matcha tea', 'sports drink', 'infused water', 'almond milk', 'soy milk', 'oat milk', 'peach juice', 'pear juice', 'pomegranate juice', 'rhubarb juice', 'soursop juice'],

    'Legumes': ['chickpeas', 'black beans', 'kidney beans', 'lentils', 'soybeans', 'pinto beans', 'navy beans', 'cannellini beans', 'lima beans', 'peanut', 'edamame', 'black-eyed peas', 'fava beans', 'adzuki beans', 'mung beans', 'split peas', 'garbanzo beans', 'green beans', 'french beans', 'white beans', 'pink beans', 'cranberry beans', 'moth beans', 'winged beans', 'runner beans', 'butter beans', 'lentil flour', 'pea protein', 'soy protein isolate'],

    'Nuts and Seeds': ['almonds', 'walnuts', 'sunflower seeds', 'pumpkin seeds', 'flaxseeds', 'chia seeds', 'sesame seeds', 'hemp seeds', 'cashews', 'pecans', 'pine nuts', 'macadamia nuts', 'pistachios', 'peanuts', 'brazil nuts', 'chestnuts', 'hazelnuts', 'quinoa', 'poppy seeds', 'sunflower butter', 'pumpkin seed butter', 'hazelnut butter', 'walnut butter', 'sunflower seed butter', 'nut butters', 'seed oils'],

    'Herbs and Spices': ['basil', 'thyme', 'rosemary', 'oregano', 'parsley', 'cilantro', 'mint', 'dill', 'sage', 'chives', 'cinnamon', 'cumin', 'coriander', 'paprika', 'turmeric', 'ginger', 'nutmeg', 'cardamom', 'vanilla', 'cayenne pepper', 'sumac', 'marjoram', 'tarragon', 'bay leaves', 'fennel', 'lavender', 'anise', 'star anise', 'black pepper', 'white pepper', 'pink peppercorns'],

    'Condiments and Sauces': ['ketchup', 'mustard', 'mayonnaise', 'soy sauce', 'barbecue sauce', 'hot sauce', 'ranch dressing', 'salsa', 'hummus', 'pesto', 'tzatziki', 'teriyaki sauce', 'vinaigrette', 'sriracha', 'honey', 'maple syrup', 'tahini', 'hoisin sauce', 'wasabi', 'chutney', 'sambal', 'mirin', 'fish sauce', 'apple cider vinegar', 'aioli', 'alfredo sauce', 'balsamic glaze', 'buffalo sauce', 'caesar dressing'],

    'Cereals': ['corn flakes', 'oatmeal', 'wheat bran', 'granola', 'rice cereal', 'bran flakes', 'shredded wheat', 'muesli', 'puffed rice', 'puffed wheat', 'multigrain cereal', 'cocoa puffs', 'rice Krispies', 'cheerios', 'special K', 'frosted flakes', 'raisin bran', 'quinoa flakes', 'millet flakes', 'amaranth flakes', 'buckwheat flakes', 'teff flakes', 'spelt flakes', 'barley flakes', 'cornmeal flakes', 'oat bran', 'wheat germ'],

    'Seafood': ['salmon', 'shrimp', 'tuna', 'tilapia', 'cod', 'trout', 'catfish', 'sardines', 'clams', 'mussels', 'oysters', 'lobster', 'crab', 'scallops', 'squid', 'anchovies', 'octopus', 'swordfish', 'haddock', 'sea bass', 'mahi-mahi', 'halibut', 'grouper', 'rockfish', 'snapper', 'caviar', 'eel', 'smoked salmon', 'whitefish', 'barramundi', 'bluefish', 'bonito', 'branzino', 'carp', 'dungeness crab', 'flounder', 'king crab', 'lake trout', 'langoustine', 'monkfish', 'mullet', 'perch', 'pike', 'pollock', 'pompano', 'rainbow trout', 'red snapper', 'sea bream', 'sea cucumber', 'sea urchin', 'shark', 'skate', 'sole', 'sturgeon', 'turbot', 'wahoo', 'yellowtail', 'yellowfin tuna']
}
food_groups_dict_updated= {
    'Fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'kiwi', 'watermelon', 'pineapple', 'mango', 'pear', 'blueberry', 'raspberry', 'blackberry', 'peach', 'plum', 'cherry', 'pomegranate', 'fig', 'cantaloupe', 'dragon fruit', 'guava', 'kiwifruit', 'passion fruit', 'cranberry', 'nectarine', 'apricot', 'papaya', 'lychee', 'star fruit (carambola)', 'persimmon', 'durian', 'jackfruit', 'tamarind', 'soursop (graviola)', 'ugli fruit (unique fruit)', 'kumquat'],

    'Vegetables': ['carrot', 'broccoli', 'spinach', 'tomato', 'cucumber', 'bell pepper', 'zucchini', 'sweet potato', 'kale', 'asparagus', 'celery', 'cauliflower', 'green beans', 'peas', 'radish', 'eggplant', 'cabbage', 'lettuce', 'onion', 'garlic', 'artichoke', 'leek', 'squash', 'brussels sprouts', 'collard greens', 'beets', 'turnip', 'rutabaga', 'okra', 'swiss chard', 'bok choy', 'parsnip', 'ginger', 'fennel', 'endive', 'arugula', 'mustard greens', 'jicama', 'kohlrabi', 'daikon', 'chicory', 'watercress', 'yam', 'horseradish', 'chayote', 'pumpkin', 'mushrooms'],

    'Red Meat': ['beef', 'pork', 'lamb', 'venison', 'bison', 'veal', 'goat', 'rabbit'],
    
    'Fast Foods': ['hamburger', 'pizza', 'french fries', 'hot dog', 'chicken nuggets', 'tacos', 'burritos', 'fried chicken', 'onion rings', 'mozzarella sticks', 'cheeseburger','gyro', 'kebab', 'quesadilla', 'chicken wings', 'nachos', 'bagels', 'pretzel', 'sliders', 'corn dog', 'fish and chips', 'potato wedges', 'philly cheesesteak', 'chili dog', 'loaded fries', 'sub sandwich', 'falafel sandwich', 'pita wrap', 'churros', 'popcorn chicken', 'meatball sub', 'bacon cheeseburger', 'chicken sandwich', 'tater tots', 'jalapeño poppers', 'calzone', 'stromboli',  'egg rolls', 'spring rolls', 'empanadas', 'big mac', 'whopper', 'nuggets', 'mcflurry',  'chicken tenders', 'spicy chicken sandwich', 'curly fries', 'taco supreme', 'cinnabon'],

	'White Meat': ['chicken', 'turkey', 'duck', 'goose',],

	'Fish': ['salmon', 'tuna', 'trout', 'halibut', 'tilapia', 'catfish', 'swordfish', 'barramundi', 'cod', 'haddock', 'mackerel', 'sardines', 'anchovies', 'mahi-mahi', 'sea bass', 'grouper', 'snapper', 'flounder', 'sole', 'pollock', 'perch', 'pike', 'carp', 'eel', 'bluefish', 'rockfish', 'wahoo', 'yellowtail', 'yellowfin tuna', 'sea bream',],

	'Other Proteins': ['tofu', 'tempeh', 'seitan', 'quorn', 'eggs', 'shrimp', 'crab', 'lobster', 'mussels', 'oysters', 'scallops', 'squid (calamari)', 'octopus', 'clams', 'peas', 'edamame', 'chickpeas', 'black beans', 'kidney beans', 'navy beans', 'pinto beans', 'soybeans', 'almonds', 'walnuts', 'cashews', 'pistachios', 'sunflower seeds', 'pumpkin seeds', 'chia seeds', 'flaxseeds', 'hemp seeds'],


    'Grains': ['rice','pasta','spaghetti', 'fettuccine', 'penne', 'rigatoni', 'farfalle', 'ravioli', 'tortellini', 'lasagna', 'orzo', 'linguine', 'cannelloni', 'gnocchi', 'tagliatelle', 'pappardelle', 'orecchiette', 'capellini', 'macaroni', 'fusilli', 'rotini', 'quinoa', 'oats', 'barley', 'wheat', 'bulgur', 'farro', 'couscous', 'millet', 'sorghum', 'buckwheat', 'amaranth', 'teff', 'spelt', 'rye', 'cornmeal', 'polenta', 'wild rice', 'fonio', 'triticale', 'einkorn', 'kamut', 'freekeh', 'sorghum', 'chickpea flour', 'semolina', 'chia seeds', 'quinoa flour', 'almond flour', 'coconut flour'],

    'Dairy': ['milk', 'cheese', 'yogurt', 'butter', 'eggs', 'cottage cheese', 'cream', 'cheddar', 'greek yogurt', 'feta', 'mozzarella', 'goat cheese', 'swiss cheese', 'brie', 'ricotta', 'sour cream', 'parmesan', 'havarti', 'provolone', 'blue cheese', 'gouda', 'asiago', 'colby jack', 'mascarpone', 'queso fresco', 'queso blanco', 'camembert', 'roquefort', 'stilton', 'emmental', 'gruyère', 'halloumi', 'manchego', 'monterey jack', 'munster', 'neufchâtel', 'paneer', 'pecorino', 'romano', 'smetana', 'tilsit', 'velveeta', 'burrata', 'clotted cream', 'crème fraîche', 'double cream', 'edam', 'fontina', 'jarlsberg', 'kefir', 'limburger', 'mascarpone', 'oaxaca', 'quark', 'reblochon', 'ricotta salata', 'stracciatella'],

    'Sweets': ['chocolate', 'ice cream', 'cookies', 'cake', 'candy', 'brownies', 'pie', 'donuts', 'gelato', 'macarons', 'cupcakes', 'muffins', 'caramel', 'fudge', 'pudding', 'sorbet', 'tiramisu', 'cheesecake', 'fruit sorbet', 'marshmallows', 'cotton candy', 'toffee', 'truffles', 'shortbread', 'baklava', 'churros', 'eclairs', 'pavlova', 'biscotti', 'cannoli', 'tarte tatin', 'key lime pie', 'lemon meringue pie', 'scones', 'croissants', 'danish pastry', 'profiteroles', 'flan', 'crème brûlée', 'mousse', 'opera cake', 'panna cotta', 'rugelach', 'sacher torte', 'strudel', 'swiss roll', 'angel food cake', 'pound cake', 'carrot cake', 'red velvet cake', 'baked alaska', 'banoffee pie', 'blondies', 'rice krispie treats'],

    'Fats and Oils': ['olive oil', 'avocado', 'nuts', 'butter', 'coconut oil', 'sunflower oil', 'almond butter', 'sesame oil', 'flaxseed oil', 'pumpkin seed oil', 'walnut oil', 'peanut butter', 'grapeseed oil', 'hazelnut oil', 'cashew butter', 'pistachio oil', 'macadamia nut oil', 'palm oil', 'canola oil', 'soybean oil', 'chia seeds', 'hemp oil', 'avocado oil', 'lard', 'beef tallow', 'duck fat', 'ghee'],

    'Beverages': ['water', 'tea', 'coffee', 'orange juice', 'apple juice', 'milkshake', 'smoothie', 'soda', 'lemonade', 'iced tea', 'hot chocolate', 'wine', 'beer', 'cocktail', 'mocktail', 'fruit punch', 'coconut water', 'vegetable juice', 'kombucha', 'chai latte', 'matcha tea', 'sports drink', 'infused water', 'almond milk', 'soy milk', 'oat milk', 'peach juice', 'pear juice', 'pomegranate juice', 'rhubarb juice', 'soursop juice'],

    'Legumes': ['chickpeas', 'black beans', 'kidney beans', 'lentils', 'soybeans', 'pinto beans', 'navy beans', 'cannellini beans', 'lima beans', 'peanut', 'edamame', 'black-eyed peas', 'fava beans', 'adzuki beans', 'mung beans', 'split peas', 'garbanzo beans', 'green beans', 'french beans', 'white beans', 'pink beans', 'cranberry beans', 'moth beans', 'winged beans', 'runner beans', 'butter beans', 'lentil flour', 'pea protein', 'soy protein isolate'],

    'Nuts and Seeds': ['almonds', 'walnuts', 'sunflower seeds', 'pumpkin seeds', 'flaxseeds', 'chia seeds', 'sesame seeds', 'hemp seeds', 'cashews', 'pecans', 'pine nuts', 'macadamia nuts', 'pistachios', 'peanuts', 'brazil nuts', 'chestnuts', 'hazelnuts', 'quinoa', 'poppy seeds', 'sunflower butter', 'pumpkin seed butter', 'hazelnut butter', 'walnut butter', 'sunflower seed butter', 'nut butters', 'seed oils'],

    'Herbs and Spices': ['basil', 'thyme', 'rosemary', 'oregano', 'parsley', 'cilantro', 'mint', 'dill', 'sage', 'chives', 'cinnamon', 'cumin', 'coriander', 'paprika', 'turmeric', 'ginger', 'nutmeg', 'cardamom', 'vanilla', 'cayenne pepper', 'sumac', 'marjoram', 'tarragon', 'bay leaves', 'fennel', 'lavender', 'anise', 'star anise', 'black pepper', 'white pepper', 'pink peppercorns'],

    'Condiments and Sauces': ['ketchup', 'mustard', 'mayonnaise', 'soy sauce', 'barbecue sauce', 'hot sauce', 'ranch dressing', 'salsa', 'hummus', 'pesto', 'tzatziki', 'teriyaki sauce', 'vinaigrette', 'sriracha', 'honey', 'maple syrup', 'tahini', 'hoisin sauce', 'wasabi', 'chutney', 'sambal', 'mirin', 'fish sauce', 'apple cider vinegar', 'aioli', 'alfredo sauce', 'balsamic glaze', 'buffalo sauce', 'caesar dressing'],

    'Cereals': ['corn flakes', 'oatmeal', 'wheat bran', 'granola', 'rice cereal', 'bran flakes', 'shredded wheat', 'muesli', 'puffed rice', 'puffed wheat', 'multigrain cereal', 'cocoa puffs', 'rice Krispies', 'cheerios', 'special K', 'frosted flakes', 'raisin bran', 'quinoa flakes', 'millet flakes', 'amaranth flakes', 'buckwheat flakes', 'teff flakes', 'spelt flakes', 'barley flakes', 'cornmeal flakes', 'oat bran', 'wheat germ'],

    'Seafood': ['salmon', 'shrimp', 'tuna', 'tilapia', 'cod', 'trout', 'catfish', 'sardines', 'clams', 'mussels', 'oysters', 'lobster', 'crab', 'scallops', 'squid', 'anchovies', 'octopus', 'swordfish', 'haddock', 'sea bass', 'mahi-mahi', 'halibut', 'grouper', 'rockfish', 'snapper', 'caviar', 'eel', 'smoked salmon', 'whitefish', 'barramundi', 'bluefish', 'bonito', 'branzino', 'carp', 'dungeness crab', 'flounder', 'king crab', 'lake trout', 'langoustine', 'monkfish', 'mullet', 'perch', 'pike', 'pollock', 'pompano', 'rainbow trout', 'red snapper', 'sea bream', 'sea cucumber', 'sea urchin', 'shark', 'skate', 'sole', 'sturgeon', 'turbot', 'wahoo', 'yellowtail', 'yellowfin tuna']
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_generated_text(text):
    """
    Cleans the AI response by removing stopwords, punctuation, and lemmatizing words.
    """
    words = word_tokenize(text.lower())
    
    filtered_words = [
        lemmatizer.lemmatize(word) for word in words 
        if word.isalpha() and word not in stop_words
    ]
    
    return " ".join(filtered_words)

def find_list_elements_in_string(food_groups_dict, generated_text):
    """
    Find and return food items and their categories from a dictionary of food groups 
    after cleaning the AI-generated response.
    """
    found_elements = {}
    
    # Clean the text before parsing
    cleaned_text = clean_generated_text(generated_text)
    
    for category, items in food_groups_dict.items():
        for item in items:
            # Match whole words only
            if re.search(rf'\b{item.lower()}\b', cleaned_text):
                if category not in found_elements:
                    found_elements[category] = []
                found_elements[category].append(item)
                
    return found_elements

