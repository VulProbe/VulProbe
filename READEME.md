# VulProbe

## 1. Data

We evaluate our method on a public dataset [BigVul](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset).

- **Data Preprocessing**
 
  Keep the dataset in `./src/resource/dataset/c`, then run the following command to process the dataset.
  ```bash
  python ./src/dataset_generator.py
  ```
  
## 2. Environmental Setup

- **Python Environment**
  
  We run our experiments using Python 3.11.7. In the `requirements.txt`, we provide the required Python packages. You can install them by running the following command.
  ```bash
  pip install -r requirements.txt
  ```

- **Install Tree-sitter**
  ```bash
  cd ./src/resource
  mkdir grammars
  cd grammars
  git clone https://github.com/tree-sitter/tree-sitter-c.git
  cd ..
  cd ..
  python build_grammars.py
  ``` 

## 3. Run
- **Vulnerability Detection**
  Run the following command to train the model for vulnerability detection.
  ```bash
  cd VulProbe/src/resource/firstStepModels/BERT
  bash run.sh
  ```

- **Train the Probe Model**
  ```bash
  cd VulProbe
  python run.py
  ```

- **Get the Preorder Traversal Representation of the AST**
  ```bash
  python customize_ast.py
  ```

- **Probe the Hidden Representation to Recover the AST and Score**
  ```bash
  python do_explain.py
  ```

- **Evaluate the VulProbe**
  ```bash
  python test.py
  ```
