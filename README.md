# Metabolomics Feature Selection System

A comprehensive feature selection system designed specifically for metabolomics data analysis. This system implements multiple feature selection methods and combines them in a pipeline to identify the most relevant metabolites for analysis.

## Project Structure

```
project/
├── core/
│   └── fs/              # Core Feature Selection functionality
├── pipeline/            # Selection Pipeline implementation
├── selection_methods/   # Individual feature selection methods
├── data/                # Data directory
├── utils/               # Utility functions
└── main.py              # Main entry point
```

## Features

### Feature Selection Methods

The system implements the following feature selection methods:

1. **Lasso Method**
   - Uses L1 regularization for feature selection
   - Configurable alpha parameter and maximum iterations

2. **Adaptive Lasso Method**
   - Enhanced version of Lasso with adaptive weights
   - Improved feature selection for correlated features

3. **Elastic Net Method**
   - Combines L1 and L2 regularization
   - Configurable alpha and l1_ratio parameters

4. **MLRRF Method**
   - Multi-Layer Random Forest based feature selection
   - Uses random state for reproducibility

5. **SVM-RFE Method**
   - Support Vector Machine Recursive Feature Elimination
   - Effective for high-dimensional data

6. **Boruta Method**
   - All-relevant feature selection
   - Identifies all features that are relevant for the target

7. **ReliefF Method**
   - Instance-based feature selection
   - Configurable number of neighbors

### Pipeline System

The project includes a flexible pipeline system that:
- Combines multiple feature selection methods
- Handles data preprocessing and transformation
- Manages feature selection process
- Provides results aggregation

### Core Functionality

The core FS class provides:
- Data loading and preprocessing
- Batch processing
- Feature selection orchestration
- Results extraction and filtering

## Requirements

The project requires the following main dependencies:
- Python 3.x
- NumPy
- Pandas
- scikit-learn
- Boruta
- skrebate
- matplotlib
- PySide6

For detailed requirements, see `requirements.txt`.

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd feature-selection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Prepare your data in Excel format with the following columns:
   - Batch information
   - Type (N for normal, other for abnormal)
   - State (will be automatically generated)
   - Feature columns (metabolites)

2. Run the feature selection:
```python
from core.fs import FS
import pandas as pd

# Load your data
data = pd.read_excel('path/to/your/data.xlsx')

# Initialize the feature selector
feature_selector = FS(X_train, X_test)

# Perform feature selection
selected_features_train, selected_features_test = feature_selector.select_features()
```

### Advanced Usage

You can customize the feature selection process by:
- Adjusting method parameters
- Modifying the pipeline configuration
- Changing the base metabolites panel
- Adjusting batch processing parameters

## Example

```python
from core.fs import FS
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Load data
data = pd.read_excel('data/discovery_set.xlsx')

# Split data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
X = data.drop(['Batch', 'type', 'batch', 'state'], axis=1)
y = data['state']

for train_index, test_index in split.split(X, y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

# Initialize and run feature selection
feature_selector = FS(X_train, X_test)
X_train_selected, X_test_selected = feature_selector.select_features()

# Save results
X_train_selected.to_csv('results/selected_features_train.csv')
X_test_selected.to_csv('results/selected_features_test.csv')
```

## Output

The system provides:
- Selected features for training and testing sets
- Logging information about the selection process
- CSV files with the selected features
- Information about the number of features before and after selection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any feedback, feel free to reach out.

[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mahdirafati680@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahdi-rafati-97420a197/)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@mehdirt)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/mahdirafati)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/itsmehdirt)