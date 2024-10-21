class SelectionPipeline:
    def __init__(self):
        self.methods = []
        self.method_metas = {}

    def add_method(self, method):
        self.methods.append(method)
    
    def apply(self, X, y):
        for method in self.methods:
            method.fit(X, y)
            self.method_metas[f"{method.__class__.__name__}"] = method.mb_
        
        return self.method_metas
