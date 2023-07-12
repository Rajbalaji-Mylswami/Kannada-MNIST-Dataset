import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Disable the warning about the use of st.pyplot() without arguments
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_data():
    # Load the .npz file
    data = np.load(r"C:\Users\shrib\Downloads\archive (4)\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\X_kannada_MNIST_train.npz")

    # Access the arrays inside the .npz file
    X_train = data['arr_0']

    # Load the .npz file
    data = np.load(r"C:\Users\shrib\Downloads\archive (4)\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\X_kannada_MNIST_test.npz")

    # Access the arrays inside the .npz file
    X_test = data['arr_0']

    # Load the .npz file
    data = np.load(r"C:\Users\shrib\Downloads\archive (4)\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\y_kannada_MNIST_train.npz")

    # Access the arrays inside the .npz file
    y_train = data['arr_0']

    # Load the .npz file
    data = np.load(r"C:\Users\shrib\Downloads\archive (4)\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\y_kannada_MNIST_test.npz")

    # Access the arrays inside the .npz file
    y_test = data['arr_0']

    return X_train, X_test, y_train, y_test


def apply_pca(X_train, X_test, components):
    # Reshape X_train to 2 dimensions
    X_train_2d = X_train.reshape(X_train.shape[0], -1)

    # Apply PCA with required components
    pca = PCA(n_components=components)
    X_train_pca = pca.fit_transform(X_train_2d)

    # Apply PCA with required components on X_test
    X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))

    return X_train_pca, X_test_pca


def train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test, model_name):
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Support Vector Machines":
        model = SVC(kernel='rbf', probability=True)
    elif model_name == "Naive Bayes":
        nb = GaussianNB()
        model = OneVsRestClassifier(nb)
    else:
        return None

    # Train the model
    model.fit(X_train_pca, y_train)

    # Make predictions on the training and test data
    y_pred_train = model.predict(X_train_pca)
    y_pred_test = model.predict(X_test_pca)

    # Compute and print precision, recall, F1-score
    report = classification_report(y_test, y_pred_test, output_dict=True)
    st.subheader("Classification Report:")
    st.write(pd.DataFrame(report).transpose())

    # Compute and plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    cm_matrix = pd.DataFrame(data=cm)
    st.subheader("Confusion Matrix:")
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    st.pyplot()

    if hasattr(model, "predict_proba"):
        # Binarize the true labels
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

        # Compute the predicted probabilities for each class
        y_scores = model.predict_proba(X_test_pca)

        # Compute the false positive rate, true positive rate, and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(10):  # num_classes is the number of classes in your dataset
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute the macro-average ROC curve and AUC
        fpr["macro"], tpr["macro"], _ = roc_curve(y_test_binarized.ravel(), y_scores.ravel())
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot ROC curves for each class and the macro-average curve
        fig, ax = plt.subplots()
        colors = ['blue', 'red', 'green', 'orange', 'purple']  # Add more colors as needed
        for i in range(10):
            if i < len(colors):
                ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                         label='ROC curve (area = %0.2f)' % roc_auc[i])
            else:
                ax.plot(fpr[i], tpr[i], lw=2,
                         label='ROC curve (area = %0.2f)' % roc_auc[i])

        ax.plot(fpr["macro"], tpr["macro"], color='black', lw=2,
                 label='Macro-average ROC curve (area = %0.2f)' % roc_auc["macro"])

        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.subheader("ROC Curve:")
        st.pyplot(fig)


def main():
    # Set title and description
    st.title("Machine Learning Model Evaluation")
    st.subheader("Kannada MNIST Dataset")

    # Load the data
    X_train, X_test, y_train, y_test = load_data()

    # Create a dropdown for selecting the number of components in PCA
    components = st.selectbox("Choose a number for components in PCA", [10, 15, 20, 25, 30])

    # Apply PCA with selected components
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, components)

    # Create a dropdown for selecting the model
    model_name = st.selectbox("Choose a model",
                              ["Decision Tree", "K-Nearest Neighbors", "Random Forest", "Support Vector Machines", "Naive Bayes"])

    # Train and evaluate the selected model
    train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test, model_name)


if __name__ == "__main__":
    main()
