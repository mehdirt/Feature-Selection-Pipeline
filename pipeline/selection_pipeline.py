class SelectionPipeline:
    def __init__(self):
        self.methods = []
        self.method_metas = {}
        self.exclude_metas = ['Glycerate-2P_Glycerate-3P_neg-006','Citraconic acid_neg-025','Pyridoxine_pos-137','Argininosuccinic acid_pos-039']

    def add_method(self, method):
        self.methods.append(method)
    
    def apply(self, X, y):
        for method in self.methods:
            method.fit(X, y)
            self.method_metas[f"{method.__class__.__name__}"] = [met for met in method.mb_ if met not in self.exclude_metas]
        
        return self.method_metas
