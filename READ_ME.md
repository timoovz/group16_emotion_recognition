# Detecting emotions with ML algorithms
The goal of this code is to recognize facial expressions from laptop camera feed. It identifies the face, makes face pictures, extracts features from the images and uses a machine learning algorithm to classify the expression on the image.

To recompute all features, empty the `data/features/` folder first.

### Data preparation
Images are 48x48 grayscale, and labels are angry, disgust, fear, happy, sad, surprise, and neutral.
- Normalize pixel values from [0,255] to [0,1]  
- Handle class imbalance by augmenting the smaller classes until all classes have the same number of samples  
- Data is augmented with:
  - translation  
  - rotation  
  - scaling  
  - lighting changes  
  - horizontal flip  
- The model is made invariant to translation, rotation, lighting changes, and horizontal flipping  
- Class weights are computed after augmentation, but not used because augmentation balances classes

### Feature engineering
- Extracted features:  
  - HOG  
  - LBP  
  - GLCM (Haralick features)  
  - DCT (low-frequency block)  
  - Haar wavelet energies (multi-scale)  
  - Hu moments  
  - Zernike moments  
- Features are stored under `data/features/train`, `data/features/validation`, and `data/features/test`  
- Feature selection steps:  
  - Variance threshold (remove features with almost no variation)  
  - Fisher score (keep the most informative features)  
  - Correlation pruning (for correlated features, keep the one with the highest mutual information)  
- Optionally, scale features using StandardScaler (zero mean and unit variance)  
- Latent representation built with PCA to reduce noise and dimensionality  

### Machine learning implementation
- **SVM**  
  - Features: HOG, LBP, DCT  
  - Selection: variance threshold > Fisher score > correlation pruning  
  - PCA to 20 components  
  - Trained with GridSearchCV and Stratified K-Fold cross-validation  
- **MLP**  
  - Features: HOG, LBP, DCT, wavelet, Hu, Zernike  
  - Selection: variance threshold > Fisher score > correlation pruning  
  - PCA keeping 99% of the variance  
  - Uses early stopping  
- **Random Forest**  
  - Features: LBP, wavelet, Hu  
  - Selection: variance threshold > Fisher score > correlation pruning  
  - PCA keeping 99% of the variance  
  - Uses class weights to handle imbalance  
- **Fuzzy system**  
  - Features: wavelet, Hu  
  - Selection: variance threshold > Fisher score > correlation pruning  
  - No PCA (low-dimensional features)  
- Evaluation metrics: accuracy, balanced accuracy, F1 score (macro), Cohen’s kappa, confusion matrix  



### Images from laptop camera feed

getting live feed is done with (OpenCV or Dlib)

### Integration