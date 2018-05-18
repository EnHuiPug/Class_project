from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from sklearn.model_selection import learning_curve

def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions = baseline.test(data.devset)

    gold_labels = [sent['gold_label'] for sent in data.devset]

    report_score(gold_labels, predictions)
    #
    # X= array[0]
    # y= array[1]
    #
    # def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
    #                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    #
    #     plt.figure()
    #     plt.title(title)
    #     if ylim is not None:
    #         plt.ylim(*ylim)
    #     plt.xlabel("Training examples")
    #     plt.ylabel("Score")
    #     train_sizes, train_scores, test_scores = learning_curve(
    #         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    #     train_scores_mean = np.mean(train_scores, axis=1)
    #     train_scores_std = np.std(train_scores, axis=1)
    #     test_scores_mean = np.mean(test_scores, axis=1)
    #     test_scores_std = np.std(test_scores, axis=1)
    #     plt.grid()
    #
    #     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                      train_scores_mean + train_scores_std, alpha=0.1,
    #                      color="r")
    #     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    #     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
    #              label="Training score")
    #     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
    #              label="Cross-validation score")
    #
    #     plt.legend(loc="best")
    #     return plt
    #
    # title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # # SVC is more expensive so we do a lower number of CV iterations:
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # estimator = SVC()
    # plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
    #
    # plt.show()


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')


