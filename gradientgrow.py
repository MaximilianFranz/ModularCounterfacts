import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm


class Decision:
    def __init__(self, dataset, chosen_attr, instance, clf):
        """
        Creates the decision object

        Args:
            dataset:
            chosen_attr:
            instance:
            clf:
        """
        self.chosen_attr = chosen_attr
        self.instance = list(instance)
        self.clf = clf
        self.dataset = np.array(dataset)
        self.last_instance = instance

        self.GSP = []
        self.walk_step = []
        self.attr_range = []
        self.difference = []

        # Defines the steps on the unit circle which to query
        self.decision_list = [0, 0.5*np.pi, np.pi, 1.5*np.pi]

        # Define the value ranges for the chosen attributes
        for item in self.chosen_attr:
            self.attr_range.append([min(self.dataset[:,item]),
                                    max(self.dataset[:,item])])

    def get_explainer(self):
        return self.clf_own

    def get_name(self):
        return 'gradient grow'


    def get_last_instance(self):
        """

        Returns: the last instance of the GradientSearchPath

        """
        dummy = np.array(self.instance)
        dummy[self.chosen_attr] = self.last_instance[1:3]
        return dummy

    def get_possible_candidates(self, count, angle, factor=1):
        """
        Selects point on the unit_circle around the instance at position count in GSP

        Args:
            count: position in the GSP which
            angle: variation of the decisions
            factor: factor by which to scale the circle, used in search_far

        Returns: list of possible candidates around the instance specified through count

        """
        poss_cands = []
        for i in range(len(self.decision_list)):
            coord1 = self.GSP[count][1] + np.cos(self.decision_list[i] + angle) * self.walk_step[0] * factor
            coord2 = self.GSP[count][2] + np.sin(self.decision_list[i] + angle) * self.walk_step[1] * factor

            dummy = np.array(self.instance)
            dummy[self.chosen_attr] = [coord1, coord2]
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]

            poss_cands.append([i, coord1, coord2, coord3])

        return poss_cands

    def filter_possible_candidates(self, possible_candidates):
        """
        Filters out edge cases in possible candidates

        Cases which are outside the boundaries of the attribute
        range of the dataset are thrown out

        Args:
            possible_candidates: possible candidates to be filtered

        Returns: list of candidates that fit the requirements
        """
        for i in range(len(possible_candidates) - 1, -1, -1):
            # traverse backwards
            if((possible_candidates[i][1] < self.attr_range[0][0])
                    or (possible_candidates[i][1] > self.attr_range[0][1])
                    or (possible_candidates[i][2] < self.attr_range[1][0])
                    or (possible_candidates[i][2] > self.attr_range[1][1])):
                        # pop candidates outside the attribute range
                        possible_candidates.pop(i)

        return possible_candidates

    def choose_one_candidate(self, possible_candidates):
        """
        Selects the best out of possible candidates if needed
        Args:
            possible_candidates: list of possible candidates

        Returns: the best candidate

        """
        if len(possible_candidates) is 1:
            return possible_candidates[0]
        else:
            # Choose candidate which maximizes prediction probability
            possible_candidates = np.array(possible_candidates)[np.array(possible_candidates)[:,3]
                                                                == max(np.array(possible_candidates)[:,3])]

            # If multiple candidates meet the maximum criteria, choose one randomly
            return possible_candidates[np.random.choice(len(possible_candidates), 1)][0]


    def search_far(self, count, last_prediction, scale):
            """
            HELPER: Scales the search sphere if no better prediction is found

            search_far is used if the four points considered in the normal
            gradientSearch do not improve the results. The factor is increased
            by 'scale' every round until a better point is found.

            Args:
                count: Current count, where to insert in the GSP
                last_prediction: last prediction in normal search path
                scale: size of increase per step

            Returns:
                chosen:
            """
            current_pred = np.round(last_prediction, decimals=3)
            factor = 1
            chosen = None
            while ((last_prediction >= current_pred) & (current_pred <= 0.5)):
                # While current prediction is worse than last_pred
                factor = factor + scale
                angle = np.random.uniform(-0.25*np.pi, 0.25*np.pi)
                poss_cands = self.get_possible_candidates(count, angle, factor)

                poss_cands = self.filter_possible_candidates(poss_cands)

                x = np.array(poss_cands)[np.array(poss_cands)[:,3] == max(np.array(poss_cands)[:,3])]
                poss_cands = [list(x[i]) for i in range(0,len(x))]
                chosen = poss_cands[np.random.choice(range(0, len(poss_cands)))]
                print("-->", factor, chosen)

                current_pred = np.round(chosen[3], decimals=3)
            return chosen


    ###########################################################################
    # gradientSearch ##########################################################
    ###########################################################################
    def gradient_search(self, step=0.01, scale=0.5, nsample=50):
        """
        Perfoms the GradientSearch step of GradientGrow

        The Goal is to find an adversarial instance by following a path toward
        prediction > 0.5.
        This is done by querying 4 instances on the unit circle around the start instance.
        The best instance is choosen and the same procedure is performed from there. All steps
        are stored in the GSP class variable.

        Args:
            step: step size relative to attribute dimension
            scale: scale to use in search_far
            nsample: number of samples
        """

        # Prep walk step sizes
        for item in self.attr_range:
            self.walk_step.append((item[1] - item[0])*step)

        # Define decision points as intervals on the unit circle in spherical coords
        decision_list = [0, 0.5*np.pi, np.pi, 1.5*np.pi]

        # set counter and check variable, which indicates whether a
        # positive example has been found
        count = 0
        check = True

        # add the original instance to the Search path with its probability
        last_prediction = np.array(self.clf.predict_proba(np.array(self.instance).reshape(1, -1))[0])[1]
        self.GSP.append([count,
                        self.instance[self.chosen_attr[0]],
                        self.instance[self.chosen_attr[1]],
                        last_prediction])

        while check:
            angle = np.random.uniform(-0.25*np.pi, 0.25*np.pi)

            possible_candidates = self.get_possible_candidates(count, angle)
            possible_candidates = self.filter_possible_candidates(possible_candidates)
            choice = self.choose_one_candidate(possible_candidates)
            print(choice)

            if choice[3] <= last_prediction:
                # If no better choice found, use search_far
                choice = self.search_far(count, last_prediction=last_prediction, scale=scale)
                if choice is None:
                    choice = self.choose_one_candidate(possible_candidates)

            count += 1
            choice[0] = count
            self.GSP.append(choice)
            print('Choice: ', choice)
            if choice[3] > 0.5 :
                check = False

            self.last_instance = choice
            last_prediction = choice[3]



    ###########################################################################
    # SectorSearch ############################################################
    ###########################################################################
    def sector_search(self, fineness=50):
        """

        Args:
            fineness (int) : specifies how small the steps are, bigger equals finer.

        Returns:
            void

        """

        # Prepare direction vectors v1, v2, v3 between last element, first element and next to last element in GSP
        v1 = (np.array(self.GSP)[0] - np.array(self.GSP)[-1])[1:3]
        v2 = (np.array(self.GSP)[-2] - np.array(self.GSP)[-1])[1:3]
        v3 = v2 - v1

        # v1 = [np.array(self.GSP)[0, 1] - np.array(self.GSP)[-1, 1],
        #       np.array(self.GSP)[0, 2] - np.array(self.GSP)[-1, 2]]
        # v2 = [np.array(self.GSP)[-2, 1] - np.array(self.GSP)[-1, 1],
        #       np.array(self.GSP)[-2, 2] - np.array(self.GSP)[-1, 2]]
        # v3 = [v2[0] - v1[0], v2[1] - v1[1]]

        v1 = [v1[0]/fineness, v1[1]/fineness]
        v2 = [v2[0]/fineness, v2[1]/fineness]
        v3 = [v3[0]/fineness, v3[1]/fineness]

        self.SSP = []
        check = 0
        factor = 1
        while (check < 6):
            rand = 0 # np.random.uniform(-v3[0]/8, v3[0]/8) LATER?
            for i in range(1, 5):
                dummy = list(self.instance)
                dummy[self.chosen_attr[0]] = np.array(self.GSP)[-1,1] + (v1[0] + v3[0]*(i*2-1)/8 + rand)*(factor + (i==2) + (i==3))
                dummy[self.chosen_attr[1]] = np.array(self.GSP)[-1,2] + (v1[1] + v3[1]*(i*2-1)/8 + rand)*(factor + (i==2) + (i==3))
                pred = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
                check = check + (pred < 0.5)
                self.SSP.append([i, dummy[self.chosen_attr[0]], dummy[self.chosen_attr[1]], pred])
            factor = factor + 2

        #--- Visualize the SectorSearch-Path
        data = np.array(self.SSP)[:,1:3]
        label = np.array(self.SSP)[:,3]

        Xgreen = np.array(data)[(np.array(label) >= 0.5)]
        Xred = np.array(data)[(np.array(label) < 0.5)]

        # plt.scatter(np.array(self.GSP)[-1,1], np.array(self.GSP)[-1,2], s=100, color='black', marker='X')
        # plt.scatter(np.array(self.GSP)[-2,1], np.array(self.GSP)[-2,2], s=50, color='blue', marker='X')
        # plt.scatter(list(np.array(Xgreen)[:,0]), list(np.array(Xgreen)[:,1]), s=40, color='green', marker='x')
        # plt.scatter(list(np.array(Xred)[:,0]), list(np.array(Xred)[:,1]), s=40, color='red', marker='x')
        # plt.scatter([self.instance[self.chosen_attr[0]]], [self.instance[self.chosen_attr[1]]], s=100, c='blue', marker='X')
        # plt.title("Ergebnisse aus SectorSearch")
        # plt.draw()



    ###########################################################################
    # svmLocal ################################################################
    ###########################################################################
    def svmLocal(self, nsample=20):
        #--- Verwende nur die letzten 15 Punkte aus dem Searchpath (falls vorhanden), um eine gute lokale Umgebung zu finden, wo wir svmQuick anwenden wollen.
        last_points_from_search_path = np.max([0,len(self.SSP)-15])

        min1 = np.min(np.array(self.SSP)[last_points_from_search_path:len(self.SSP),1])
        max1 = np.max(np.array(self.SSP)[last_points_from_search_path:len(self.SSP),1])
        min2 = np.min(np.array(self.SSP)[last_points_from_search_path:len(self.SSP),2])
        max2 = np.max(np.array(self.SSP)[last_points_from_search_path:len(self.SSP),2])
        self.decisionRange = [min1, max1, min2, max2]

        sample = []
        for i in range(0, nsample):
            coord1 = np.random.uniform(min1, max1)
            coord2 = np.random.uniform(min2, max2)

            dummy = list(self.instance)
            dummy[self.chosen_attr[0]] = coord1
            dummy[self.chosen_attr[1]] = coord2
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]

            sample.append([i, coord1, coord2, coord3])

        #--- Find Support Vector Machine
        X = np.array(sample)[:,1:3]
        y = np.array(sample)[:,3]
        for i in range(0,len(y)):
            y[i] = int(y[i] >= 0.5)
        y = np.array(y)

        #--- Skalieren der Daten
        scale_min1,scale_max1 = np.min(X[:,0]),np.max(X[:,0])
        scale_min2,scale_max2 = np.min(X[:,1]),np.max(X[:,1])

        for j in range(0,len(X)):
            X[j,0] = (X[j,0] - scale_min1) / (scale_max1 - scale_min1)
            X[j,1] = (X[j,1] - scale_min2) / (scale_max2 - scale_min2)

        #--- create svm
        clf_svm = svm.SVC(kernel='linear', C=10.0, tol=1e-5, max_iter=-1)
        clf_svm.fit(X,y)
        self.clf_own = clf_svm

        #--- create a mesh to plot
        x_min, x_max = min(X[:, 0]), max(X[:, 0])
        y_min, y_max = min(X[:, 1]), max(X[:, 1])
        hx = (x_max - x_min)/100
        hy = (y_max - y_min)/100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

        #--- draw decision border svm
        w1 = clf_svm.coef_[0]
        self.svmQuick_m = -w1[0] / w1[1] * (scale_max2 - scale_min2)/(scale_max1 - scale_min1)
        self.svmQuick_c = -clf_svm.intercept_[0] / w1[1] * (scale_max2 - scale_min2) + scale_min2 - scale_min1*self.svmQuick_m
        x_line = np.linspace(x_min*(scale_max1 - scale_min1)+scale_min1,x_max*(scale_max1 - scale_min1)+scale_min1)
        y_line = self.svmQuick_m * x_line + self.svmQuick_c
        # plt.plot(x_line, y_line, 'k-', lw=1)


        #--- Predict the result by giving Data to the model
        Z = clf_svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        #--- Reskalieren
        for j in range(0,len(xx)):
            xx[j] = xx[j] * (scale_max1 - scale_min1) + scale_min1
            yy[j] = yy[j] * (scale_max2 - scale_min2) + scale_min2

        for j in range(0,len(X)):
            X[j,0] = X[j,0] * (scale_max1 - scale_min1) + scale_min1
            X[j,1] = X[j,1] * (scale_max2 - scale_min2) + scale_min2

        # plt.contourf(xx, yy, Z, levels=[0.0,0.5,1.0], colors=('r','g'), alpha = 0.4)
        # plt.scatter(X[:, 0], X[:, 1], c=['green'*int(y[i])+'red'*(1-int(y[i])) for i in range(0,len(y))], cmap = plt.cm.Paired, marker='s', s=10)
        # plt.xlabel("Attribut " + str(self.chosen_attr[0]))
        # plt.ylabel("Attribut " + str(self.chosen_attr[1]))
        # plt.title("Lokale Entscheidungsgrenze mit SVMQuick")
        # plt.xlim(scale_min1, scale_max1)
        # plt.ylim(scale_min2, scale_max2)
        # plt.draw()



    ###########################################################################
    # Extension ###############################################################
    ###########################################################################
    def Extension(self, limit=20):
        m = self.svmQuick_m
        c = self.svmQuick_c
        #--- SVM: y = m * x + c

        u1 = [(self.decisionRange[1] - self.decisionRange[0])/2, m*(self.decisionRange[1] - self.decisionRange[0])/2]
        u1_norm = [u1[0]/(self.decisionRange[1] - self.decisionRange[0]), u1[1]/(self.decisionRange[3] - self.decisionRange[2])]
        u1_norm2 = [u1_norm[0]/np.sqrt(u1_norm[0]*u1_norm[0] + u1_norm[1]*u1_norm[1]), u1_norm[1]/np.sqrt(u1_norm[0]*u1_norm[0] + u1_norm[1]*u1_norm[1])]
        u2_norm = [-u1_norm2[1]/u1_norm2[0], 1]
        u2 = [u2_norm[0]*(self.decisionRange[1] - self.decisionRange[0])/4, u2_norm[1]*(self.decisionRange[3] - self.decisionRange[2])/4]
        initial = [(self.decisionRange[1] + self.decisionRange[0])/2, m * (self.decisionRange[1] + self.decisionRange[0])/2 + c]

        self.eval_range = []
        self.eval_range.append(self.instance[self.chosen_attr[0]])
        self.eval_range.append(self.instance[self.chosen_attr[1]])

        #--- nach rechts
        self.result = []
        pred_curr = 5
        pred_prev = 6
        i = 0
        while((pred_curr != pred_prev) & (i <= limit)):
            coord1 = initial[0] + np.cos(i*np.pi)*u2[0] + i*u1[0]
            coord2 = initial[1] + np.cos(i*np.pi)*u2[1] + i*u1[1]

            dummy = list(self.instance)
            dummy[self.chosen_attr[0]] = coord1
            dummy[self.chosen_attr[1]] = coord2
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]

            self.result.append([i, coord1, coord2, coord3])

            pred_prev = pred_curr
            pred_curr = (coord3 >= 0.5)
            i = i + 1

        if(self.result[-1][0] >= 2):
            self.eval_range.append(self.result[-3][1])
            self.eval_range.append(self.result[-3][2])


#        if(i < limit):
#            coord1 = self.result[-1][1] - u2[0]
#            coord2 = self.result[-1][2] - u2[1]
#            dummy = list(self.instance)
#            dummy[self.chosen_attr[0]] = coord1
#            dummy[self.chosen_attr[1]] = coord2
#            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
#            self.result.append([i, coord1, coord2, coord3])
#
#            coord1 = self.result[-4][1] - u2[0]
#            coord2 = self.result[-4][2] - u2[1]
#            dummy = list(self.instance)
#            dummy[self.chosen_attr[0]] = coord1
#            dummy[self.chosen_attr[1]] = coord2
#            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]
#            self.result[-3][1] = coord1
#            self.result[-3][2] = coord2
#            self.result[-3][3] = coord3
#
#            self.borderGrowth()


        #--- nach links
        pred_curr = 5
        pred_prev = 6
        i = 0
        while((pred_curr != pred_prev) & (i >= -limit)):
            coord1 = initial[0] + np.cos(i*np.pi)*u2[0] + i*u1[0]
            coord2 = initial[1] + np.cos(i*np.pi)*u2[1] + i*u1[1]

            dummy = list(self.instance)
            dummy[self.chosen_attr[0]] = coord1
            dummy[self.chosen_attr[1]] = coord2
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]

            self.result.append([-i, coord1, coord2, coord3])

            pred_prev = pred_curr
            pred_curr = (coord3 >= 0.5)
            i = i - 1

        if(self.result[-1][0] >= 2):
            self.eval_range.append(self.result[-3][1])
            self.eval_range.append(self.result[-3][2])


        #--- nach oben
        pred_curr = 1
        pred_prev = 1
        i = 1
        while((pred_curr == pred_prev) & (i-1 <= limit)):
            coord1 = initial[0] + i*u2[0]
            coord2 = initial[1] + i*u2[1]

            dummy = list(self.instance)
            dummy[self.chosen_attr[0]] = coord1
            dummy[self.chosen_attr[1]] = coord2
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]

            self.result.append([i, coord1, coord2, coord3])

            pred_prev = pred_curr
            pred_curr = (coord3 >= 0.5)
            i = i + 1

        if(self.result[-1][0] >= 2):
            self.eval_range.append(self.result[-3][1])
            self.eval_range.append(self.result[-3][2])

        self.eval_range = np.array(self.eval_range)


        #--- Visualize points
        data = np.array(self.result)[:,1:3]
        label = np.array(self.result)[:,3]

        Xgreen = np.array(data)[(np.array(label) >= 0.5)]
        Xred = np.array(data)[(np.array(label) < 0.5)]

        # plt.scatter(np.array(self.GSP)[-1,1], np.array(self.GSP)[-1,2], s=100, color='black', marker='X')
        # plt.scatter(list(np.array(Xgreen)[:,0]), list(np.array(Xgreen)[:,1]), s=40, color='green', marker='x')
        # plt.scatter(list(np.array(Xred)[:,0]), list(np.array(Xred)[:,1]), s=40, color='red', marker='x')
        # plt.scatter([self.instance[self.chosen_attr[0]]], [self.instance[self.chosen_attr[1]]], s=100, c='blue', marker='X')
        # plt.xlabel("Attribut " + str(self.chosen_attr[0]))
        # plt.ylabel("Attribut " + str(self.chosen_attr[1]))
        # plt.title("Lokale Entscheidungsgrenze mit SVMQuick nach Extension")
        # plt.draw()



    ###########################################################################
    # borderGrowth ############################################################
    ###########################################################################
    def borderGrowth(self, nsample=200, limit=20):
        #last_points = np.array([self.result[-3], self.result[-1]])
        last_points = np.array(self.result[-4:])
        min1 = np.min(last_points[:,1])
        max1 = np.max(last_points[:,1])
        min2 = np.min(last_points[:,2])
        max2 = np.max(last_points[:,2])
        self.decisionRange = [min1, max1, min2, max2]

        sample = []
        for j in range(0, nsample):
            coord1 = np.random.uniform(min1, max1)
            coord2 = np.random.uniform(min2, max2)

            dummy = list(self.instance)
            dummy[self.chosen_attr[0]] = coord1
            dummy[self.chosen_attr[1]] = coord2
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]

            sample.append([j, coord1, coord2, coord3])

        #--- Find Support Vector Machine
        X = np.array(sample)[:,1:3]
        y = np.array(sample)[:,3]
        for j in range(0,len(y)):
            y[j] = int(y[j] >= 0.5)
        y = np.array(y)

        #--- Skalieren der Daten
        scale_min1,scale_max1 = np.min(X[:,0]),np.max(X[:,0])
        scale_min2,scale_max2 = np.min(X[:,1]),np.max(X[:,1])

        for j in range(0,len(X)):
            X[j,0] = (X[j,0] - scale_min1) / (scale_max1 - scale_min1)
            X[j,1] = (X[j,1] - scale_min2) / (scale_max2 - scale_min2)

        #--- create svm
        clf_svm = svm.SVC(kernel='linear', C=100.0, tol=1e-6, max_iter=-1)
        clf_svm.fit(X,y)
        self.clf_own = clf_svm

        #--- create a mesh to plot
        x_min, x_max = min(X[:, 0]), max(X[:, 0])
        y_min, y_max = min(X[:, 1]), max(X[:, 1])
        hx = (x_max - x_min)/100
        hy = (y_max - y_min)/100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

        #--- draw decision border
        w = clf_svm.coef_[0]
        m = -w[0] / w[1] * (scale_max2 - scale_min2)/(scale_max1 - scale_min1)
        c = -clf_svm.intercept_[0] / w[1] * (scale_max2 - scale_min2) + scale_min2 - scale_min1*m
        x_line = np.linspace(x_min*(scale_max1 - scale_min1)+scale_min1, x_max*(scale_max1 - scale_min1)+scale_min1)
        y_line = m * x_line + c
        # plt.plot(x_line, y_line, 'k-', lw=1)

        #--- Predict the result by giving Data to the model
        Z = clf_svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        #--- Reskalieren
        for j in range(0,len(xx)):
            xx[j] = xx[j] * (scale_max1 - scale_min1) + scale_min1
            yy[j] = yy[j] * (scale_max2 - scale_min2) + scale_min2

        for j in range(0,len(X)):
            X[j,0] = X[j,0] * (scale_max1 - scale_min1) + scale_min1
            X[j,1] = X[j,1] * (scale_max2 - scale_min2) + scale_min2

        # plt.contourf(xx, yy, Z, levels=[0.0,0.5,1.0], colors=('r','g'), alpha = 0.4)
        plt.scatter(X[:, 0], X[:, 1], c=['green'*int(y[i])+'red'*(1-int(y[i])) for i in range(0,len(y))], cmap = plt.cm.Paired, marker='s', s=10)
        plt.xlabel(self.dataset.feature_names[self.chosen_attr[0]])
        plt.ylabel(self.dataset.feature_names[self.chosen_attr[1]])
        plt.title("Lokale Entscheidungsgrenze mit SVMQuick - nach Bordergrowth")
        plt.xlim(scale_min1, scale_max1)
        plt.ylim(scale_min2, scale_max2)
        # plt.draw()


        #--- SVM: y = m * x + c
        u1 = [-(self.decisionRange[3] - self.decisionRange[2])/(2*m), -(self.decisionRange[3] - self.decisionRange[2])/2]
        u1_norm = [u1[0]/(self.decisionRange[1] - self.decisionRange[0]), u1[1]/(self.decisionRange[3] - self.decisionRange[2])]
        u1_norm2 = [u1_norm[0]/np.sqrt(u1_norm[0]*u1_norm[0] + u1_norm[1]*u1_norm[1]), u1_norm[1]/np.sqrt(u1_norm[0]*u1_norm[0] + u1_norm[1]*u1_norm[1])]
        u2_norm = [1, -u1_norm2[0]/u1_norm2[1]]
        u2 = [u2_norm[0]*(self.decisionRange[1] - self.decisionRange[0])/4, u2_norm[1]*(self.decisionRange[3] - self.decisionRange[2])/4]
        initial = [((self.decisionRange[3] + self.decisionRange[2])/2 - c)/m, (self.decisionRange[3] + self.decisionRange[2])/2]

        #--- nach rechts
        pred_curr = 5
        pred_prev = 6
        i = 0
        while((pred_curr != pred_prev) & (i <= limit)):
            coord1 = initial[0] + np.cos(i*np.pi)*u2[0] + i*u1[0]
            coord2 = initial[1] + np.cos(i*np.pi)*u2[1] + i*u1[1]

            dummy = list(self.instance)
            dummy[self.chosen_attr[0]] = coord1
            dummy[self.chosen_attr[1]] = coord2
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]

            self.result.append([i, coord1, coord2, coord3])

            pred_prev = pred_curr
            pred_curr = (coord3 >= 0.5)
            i = i + 1

        #--- nach oben
        pred_curr = 1
        pred_prev = 1
        i = 1
        while((pred_curr == pred_prev) & (i-1 <= limit)):
            coord1 = initial[0] + i*u2[0]
            coord2 = initial[1] + i*u2[1]

            dummy = list(self.instance)
            dummy[self.chosen_attr[0]] = coord1
            dummy[self.chosen_attr[1]] = coord2
            coord3 = np.array(self.clf.predict_proba(np.array(dummy).reshape(1, -1))[0])[1]

            self.result.append([i, coord1, coord2, coord3])

            pred_prev = pred_curr
            pred_curr = (coord3 >= 0.5)
            i = i + 1
