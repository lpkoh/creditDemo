import streamlit as st
import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle

singleOrMultiple = st.sidebar.selectbox(label='Select mode', options=['Choose your mode:', 'Single model', 'Multi model'])

if singleOrMultiple == 'Single model':
    st.title(singleOrMultiple)

    dataSelection = st.selectbox(label="", options=['Choose your dataset', 'Home Credit Dataset'])

    if dataSelection == 'Home Credit Dataset':
        st.title('Import data')
        # Import
        st.write('Model fitted on 240000 training examples')
        xTrain = pd.read_csv('./xTest.csv')
        yTrain = pd.read_csv('./yTest.csv')
        st.write('Shape of xTest, yTest')
        st.write(xTrain.shape)
        st.write(yTrain.shape)
        st.write('First five test X')
        st.write(xTrain.head())
        st.write('First five test y')
        st.write(yTrain.head())

    modelSelection = st.selectbox(label='', options=['Choose your model', 'Logistic Regression', 'Naive Bayes'])
    if modelSelection == 'Logistic Regression':
        # load the log reg model
        lr = None
        with open("lr_model.pkl", 'rb') as file:
            lr = pickle.load(file)
        lrTrainPredProb = lr.predict_proba(xTrain)[:, 1].reshape(-1, 1)
    elif modelSelection == 'Naive Bayes':
        # fit the decision tree model
        lr = None
        with open("tM_model.pkl", 'rb') as file:
            lr = pickle.load(file)
        lrTrainPredProb = lr.predict_proba(xTrain)[:, 1].reshape(-1, 1)

    if dataSelection == 'Home Credit Dataset' and (modelSelection == 'Logistic Regression' or modelSelection == 'Naive Bayes'):
        # We conduct model only analysis on log reg model
        st.title('Evaluating a single model: ' + modelSelection)
        st.markdown('### First technique: ANOVA')

        @st.cache
        def ANOVAFunc(predictedProb, threshold, y):
            ANOVACut = predictedProb > threshold
            predictedBad = y[ANOVACut]
            predictedGood = y[~ANOVACut]
            badMean = np.mean(predictedBad)
            goodMean = np.mean(predictedGood)
            badVar = np.var(predictedBad)
            goodVar = np.var(predictedGood)
            return badMean, goodMean, (badMean - goodMean) ** 2 / (0.5 * (goodVar + badVar))

        thresholdANOVA = st.slider("Probability cutoff for approval", min_value=0.0, max_value=1.0, value=0.5,
                                   step=0.01, key=1)
        lrBadMean, lrGoodMean, lrANOVAStat = ANOVAFunc(lrTrainPredProb, thresholdANOVA, yTrain)
        st.write('Mean default rate in group predicted to be bad')
        st.write(str(lrBadMean[0]))
        st.write('Mean default rate in group predicted to be good')
        st.write(str(lrGoodMean[0]))
        st.write('ANOVA statistic for difference in default rate between predicted bad and predicted good:')
        st.write(str(lrANOVAStat[0]))

        st.markdown('### Second technique: KS Statistic')

        @st.cache
        def KSFunc(predictedProb, y):
            badProb = predictedProb[y == 1]
            badProb = np.sort(badProb)
            goodProb = predictedProb[y == 0]
            goodProb = np.sort(goodProb)
            KSstat = stats.ks_2samp(badProb, goodProb)
            return badProb, goodProb, KSstat

        lrBadProb, lrGoodProb, lrKSStat = KSFunc(lrTrainPredProb, yTrain)
        # KS statistic
        st.write('KS statistic: ' + str(lrKSStat[0]))
        st.write('p-value: : ' + str(lrKSStat[1]))
        st.write('Red line: CDF of good loans')
        st.write('Blue line is CDF of bad loans')
        plt.plot(lrBadProb, np.arange(len(lrBadProb)) / float(len(lrBadProb)), color='b')
        plt.plot(lrGoodProb, np.arange(len(lrGoodProb)) / float(len(lrGoodProb)), color='r')
        st.pyplot()

        st.markdown('### Third technique: ROC Curve')

        @st.cache
        def ROCFunc(y, predictedProb):
            AUC = metrics.roc_auc_score(np.array(y['TARGET']), predictedProb)
            FPR, TPR, Thresholds = metrics.roc_curve(np.array(y['TARGET']), predictedProb, pos_label=1)
            return AUC, FPR, TPR, Thresholds


        lrAUC, lrFPR, lrTPR, lrThresholds = ROCFunc(yTrain, lrTrainPredProb)
        lw = 2
        plt.plot(lrFPR, lrTPR, lw=lw, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        st.pyplot()

        st.markdown('### Fourth technique: Gini Score')


        @st.cache
        def GiniFunc(y, predictedProb):
            return metrics.roc_auc_score(np.array(y['TARGET']), predictedProb) * 2 - 1


        st.write('Gini Score:')
        st.write(str(GiniFunc(yTrain, lrTrainPredProb)))

        st.markdown('### Odds ratio:')
        binEdges = []
        for i in range(6):
            binEdges.append(i * 0.2)
        binEdges[5] += 0.01


        @st.cache
        def OddsFunc(y, predictedProb):
            Odds = []
            for i in range(5):
                subsetY = y[(predictedProb >= binEdges[i]) & (predictedProb < binEdges[i + 1])]
                Odds.append((len(subsetY) - np.sum(subsetY)[0]) / np.sum(subsetY)[0])
            Odds.reverse()
            return Odds


        lrOdds = OddsFunc(yTrain, lrTrainPredProb)
        plt.plot(range(5), lrOdds)
        plt.xlabel('Predicted probability of not defaulting bucket')
        plt.ylabel('Odds ratio of good loans to bad loans')
        st.pyplot()

        st.markdown('### Table of performance per tranche')
        lrTrancheTable = pd.DataFrame(
            columns=['Predicted Probability of Default', 'Number of loans', 'Cumulative % of loans',
                     'Number of bad loans',
                     'Cumulative % of bad loans', 'Odds (good to bad)'])
        cumSum = 0
        cumSumNeg = 0
        for i in range(5):
            rowToAppend = []
            lowerBinEdge = round(binEdges[5 - i - 1], 2)
            upperBinEdge = round(binEdges[5 - i], 2)
            rowToAppend.append(str(lowerBinEdge) + ' to ' + str(upperBinEdge))
            subsetY = yTrain[(lrTrainPredProb >= lowerBinEdge) & (lrTrainPredProb < upperBinEdge)]
            rowToAppend.append(subsetY.shape[0])
            cumSum += subsetY.shape[0]
            rowToAppend.append(cumSum / yTrain.shape[0])
            rowToAppend.append(np.sum(subsetY)[0])
            cumSumNeg += np.sum(subsetY)[0]
            rowToAppend.append(cumSumNeg / np.sum(yTrain)[0])
            rowToAppend.append((len(subsetY) - np.sum(subsetY)[0]) / np.sum(subsetY)[0])
            rowToAppend = pd.DataFrame([rowToAppend],
                                       columns=['Predicted Probability of Default', 'Number of loans',
                                                'Cumulative % of loans',
                                                'Number of bad loans', 'Cumulative % of bad loans',
                                                'Odds (good to bad)'])
            lrTrancheTable = lrTrancheTable.append(rowToAppend)
        st.write(lrTrancheTable)
elif singleOrMultiple == 'Multi model':
    st.title(singleOrMultiple)

    dataSelection = st.selectbox(label="", options=['Choose your dataset', 'Home Credit Dataset'])

    if dataSelection == 'Home Credit Dataset':
        st.title('Import and engineer data')
        # Import
        st.write('Model fitted on 240000 training examples')
        xTrain = pd.read_csv('./xTest.csv')
        yTrain = pd.read_csv('./yTest.csv')
        st.write('Shape of xTest, yTest')
        st.write(xTrain.shape)
        st.write(yTrain.shape)
        st.write('First five test X')
        st.write(xTrain.head())
        st.write('First five test y')
        st.write(yTrain.head())

        # load the log reg model
        lr = None
        with open("lr_model.pkl", 'rb') as file:
            lr = pickle.load(file)
        lrTrainPredProb = lr.predict_proba(xTrain)[:, 1].reshape(-1, 1)

        # fit the decision tree model
        tM = None
        with open("tM_model.pkl", 'rb') as file:
            tM = pickle.load(file)
        tmTrainPredProb = tM.predict_proba(xTrain)[:, 1].reshape(-1, 1)

        modelSelection = st.multiselect(label='Choose your model', options=['Naive Bayes', 'Logistic Regression'])
        multiButton = st.button(label='Compare')
        binEdges = []
        for i in range(6):
            binEdges.append(i * 0.2)
        binEdges[5] += 0.01
        @st.cache
        def ROCFunc(y, predictedProb):
            AUC = metrics.roc_auc_score(np.array(y['TARGET']), predictedProb)
            FPR, TPR, Thresholds = metrics.roc_curve(np.array(y['TARGET']), predictedProb, pos_label=1)
            return AUC, FPR, TPR, Thresholds
        lrAUC, lrFPR, lrTPR, lrThresholds = ROCFunc(yTrain, lrTrainPredProb)
        if multiButton:
            st.title('Comparing two model: logistic regression vs naive bayes')
            @st.cache
            def scoreShiftFunc(predictProbA, predictProbB):
                scoreShiftMatrix = np.zeros(shape=(5, 5))
                for i in range(predictProbA.shape[0]):
                    rowIndex = None
                    colIndex = None
                    for j in range(5):
                        if predictProbA[i] >= binEdges[j] and predictProbA[i] < binEdges[j + 1]:
                            rowIndex = j
                        if predictProbB[i] >= binEdges[j] and predictProbB[i] < binEdges[j + 1]:
                            colIndex = j
                    scoreShiftMatrix[rowIndex, colIndex] += 1
                rowNames = ['0 to 0.2', '0.2 to 0.4', '0.4 to 0.6', '0.6 to 0.8', '0.8 to 1']
                colNames = ['0 to 0.2', '0.2 to 0.4', '0.4 to 0.6', '0.6 to 0.8', '0.8 to 1']
                scoreShiftMatrix = np.true_divide(scoreShiftMatrix, predictProbA.shape[0])
                scoreShiftMatrix = pd.DataFrame(scoreShiftMatrix, columns=colNames, index=rowNames)
                return scoreShiftMatrix


            scoreShiftMatrix = scoreShiftFunc(lrTrainPredProb, tmTrainPredProb)
            st.markdown('### Score shift')
            st.write(scoreShiftMatrix)

            st.markdown('### Side by side graphs:')
            lw = 2
            plt.subplot(1, 2, 1)
            plt.plot(lrFPR, lrTPR, lw=lw, color='darkorange')
            plt.title('ROC for logistic regression')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.subplot(1, 2, 2)
            tmAUC, tmFPR, tmTPR, tmThresholds = ROCFunc(yTrain, tmTrainPredProb)
            plt.plot(tmFPR, tmTPR, lw=lw, color='darkorange')
            plt.title('ROC for random forest')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.tight_layout()
            st.pyplot()

            st.markdown('### Side by side statistics:')


            @st.cache
            def sideBySideTableFunc(xTest, predictedProbA, predictedProbB, threshold, y):
                sidebysideColumns = ['Metric', 'Current Strategy using Score A', 'Current strategy using Score B']
                sideBySideTable = pd.DataFrame(columns=sidebysideColumns)
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Number of Applicants', xTest.shape[0], xTest.shape[0]]], columns=
                    sidebysideColumns))
                AAccepts = predictedProbA < threshold
                BAccepts = predictedProbB > threshold
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Number of Accepts', np.sum(AAccepts), np.sum(BAccepts)]], columns=
                    sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Acceptance rate', np.mean(AAccepts), np.mean(BAccepts)]], columns=
                    sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Number of Bad accounts', np.sum(y[AAccepts]['TARGET']),
                                   np.sum(y[BAccepts]['TARGET'])]], columns=sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Bad rate', np.sum(y[AAccepts]['TARGET']) / np.sum(AAccepts),
                                   np.sum(y[BAccepts]['TARGET']) / np.sum(BAccepts)]], columns=sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Estimated Loss Per Bad Account', 2500, 2500]], columns=sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Number of Good accounts', np.sum(AAccepts) - np.sum(y[AAccepts]['TARGET']),
                                   np.sum(BAccepts) - np.sum(y[BAccepts]['TARGET'])]], columns=sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Estiamted Revenue Per Good Account', 100, 100]], columns=sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Total Revenue', (np.sum(AAccepts) - np.sum(y[AAccepts]['TARGET'])) * 100,
                                   (np.sum(BAccepts) - np.sum(y[BAccepts]['TARGET'])) * 100]],
                                 columns=sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Total Loss', np.sum(y[AAccepts]['TARGET']) * 2500,
                                   np.sum(y[BAccepts]['TARGET']) * 2500]], columns=sidebysideColumns))
                sideBySideTable = sideBySideTable.append(
                    pd.DataFrame([['Total Profit', (np.sum(AAccepts) - np.sum(y[AAccepts]['TARGET'])) * 100 -
                                   np.sum(y[AAccepts]['TARGET']) * 2500,
                                   (np.sum(BAccepts) - np.sum(y[BAccepts]['TARGET'])) * 100 -
                                   np.sum(y[BAccepts]['TARGET']) * 2500]], columns=sidebysideColumns))
                profitChange = (np.sum(BAccepts) - np.sum(y[BAccepts]['TARGET'])) * 100 - np.sum(
                    y[BAccepts]['TARGET']) * 2500 - \
                               (np.sum(AAccepts) - np.sum(y[AAccepts]['TARGET'])) * 100 + np.sum(
                    y[AAccepts]['TARGET']) * 2500
                return sideBySideTable, profitChange


            acceptThreshold = st.slider("Probability cutoff for approval", min_value=0.0, max_value=1.0, value=0.5,
                                        step=0.01, key=2)
            sideBySideTable, profitChange = sideBySideTableFunc(xTrain, lrTrainPredProb, tmTrainPredProb,
                                                                acceptThreshold, yTrain)
            st.write(sideBySideTable)
            st.write('Profit increase:')
            st.write(str(profitChange))

            st.markdown('### Swap set analysis:')
            st.write('Original Breakdown:')
            swapSetThreshold = st.slider("Probability cutoff for approval", min_value=0.0, max_value=1.0, value=0.3,
                                         step=0.01, key=3)
            lrAccepts = lrTrainPredProb < swapSetThreshold
            tmAccepts = tmTrainPredProb < swapSetThreshold
            preSwapTable = np.zeros((2, 2))
            preSwapTable[0, 0] = np.sum(lrAccepts[lrAccepts == tmAccepts])
            preSwapTable[1, 1] = np.sum(~lrAccepts[lrAccepts == tmAccepts])
            preSwapTable[0, 1] = np.sum(lrAccepts[lrAccepts != tmAccepts])
            preSwapTable[1, 0] = np.sum(tmAccepts[lrAccepts != tmAccepts])
            preSwapTable = pd.DataFrame(preSwapTable, columns=['Accepted by B', 'Rejected by B'],
                                        index=['Accepted by A', 'Rejected by A'])
            st.write(preSwapTable)

            st.write('Swapped to maintain same approval rate:')
            swapSameApprovalRate = np.zeros((3, 3))
            swapSameApprovalRate[0, 0] = np.sum(lrAccepts[lrAccepts == tmAccepts])
            swapSameApprovalRate[0, 1] = np.sum(lrAccepts[lrAccepts != tmAccepts])
            swapSameApprovalRate[1, 0] = np.sum(lrAccepts[lrAccepts != tmAccepts])
            swapSameApprovalRate[1, 1] = np.sum(~lrAccepts[lrAccepts == tmAccepts])
            swapSameApprovalRate[0, 2] = np.sum(yTrain[lrAccepts]['TARGET']) / np.sum(lrAccepts)
            swapIn = yTrain[(lrAccepts != tmAccepts) & (tmAccepts == 1)].sample(
                n=np.sum(lrAccepts[lrAccepts != tmAccepts]))
            swapSameApprovalRate[2, 0] = (np.sum(
                yTrain[(lrAccepts == tmAccepts) & (lrAccepts == 1)]['TARGET']) + np.sum(swapIn['TARGET'])) / \
                                         (swapSameApprovalRate[1, 0] + swapSameApprovalRate[0, 0])
            swapSameApprovalRate = pd.DataFrame(swapSameApprovalRate,
                                                columns=['Accepted by B', 'Rejected by B', 'Old Bad Rate'],
                                                index=['Accepted by A', 'Rejected by A', 'New Bad Rate'])
            st.write('Green is new bad rate')
            st.write('Red is old bad rate')


            def highlight_cells(x):
                df = x.copy()
                # set default color
                # df.loc[:,:] = 'background-color: papayawhip'
                df.loc[:, :] = ''
                # set particular cell colors
                df.iloc[0, 2] = 'background-color: red'
                df.iloc[2, 0] = 'background-color: lightgreen'
                return df


            swapSameApprovalRate = swapSameApprovalRate.style.apply(highlight_cells, axis=None)
            st.write(swapSameApprovalRate)
else:
    st.title('Please choose a mode')