import json

import torchvision


# DATA FILTERS
class WebdatasetFilter():
    def __init__(self, min_size=512, max_size=4096, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99, text_conditions=None): # {'min_words': 2, 'forbidden_words': ["www.", ".com", "http", "-", "_", ":", ";", "(", ")", "/", "%", "|", "?", "download", "interior", "kitchen", "chair", "getty", "how", "what", "when", "why", "laminate", "furniture", "hair", "dress", "clothing"]}):
        self.min_size = min_size
        self.max_size = max_size
        self.max_pwatermark = max_pwatermark
        self.aesthetic_threshold = aesthetic_threshold
        self.unsafe_threshold = unsafe_threshold
        self.text_conditions = text_conditions 

    def __call__(self, x):
        try:
            if 'json' in x:
                x_json = json.loads(x['json'])
                filter_size_min = (x_json.get('original_width', 0.0) or 0.0) >= self.min_size and x_json.get('original_height', 0) >= self.min_size
                filter_size_max = (x_json.get('original_width', 0.0) or 0.0) <= self.max_size and x_json.get('original_height', 0) <= self.max_size
                filter_watermark = (x_json.get('pwatermark', 1.0) or 1.0) <= self.max_pwatermark
                filter_aesthetic_a = (x_json.get('aesthetic', 0.0) or 0.0) >= self.aesthetic_threshold
                filter_aesthetic_b = (x_json.get('AESTHETIC_SCORE', 0.0) or 0.0) >= self.aesthetic_threshold
                filter_unsafe = (x_json.get('punsafe', 1.0) or 1.0) <= self.unsafe_threshold
                if self.text_conditions is not None:
                    caption = x['txt'].decode("utf-8") 
                    filter_min_words = len(caption.split(" ")) >= self.text_conditions['min_words']
                    filter_ord_128 = all([ord(c) < 128 for c in caption])
                    filter_forbidden_words = all([c not in caption.lower() for c in self.text_conditions['forbidden_words']])
                    filter_text = filter_min_words and filter_ord_128 and filter_forbidden_words
                else:
                    filter_text = True
                return filter_size_min and filter_size_max and filter_watermark and (filter_aesthetic_a or filter_aesthetic_b) and filter_unsafe and filter_text
            else:
                return False
        except:
            return False


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(512),
    torchvision.transforms.RandomCrop(512),
])

effnet_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(384, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    torchvision.transforms.CenterCrop(384),
    torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
])


def identity(x):
    return x