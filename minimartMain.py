import matplotlib.pyplot as plt
import numpy as np
from amplpy import AMPL, Environment
from matplotlib import cm

# retrieve data from AMPL model

ampl = AMPL(Environment('/home/luisa/ampl_linux-intel64'))

ampl.set_option('solver', 'cplex')

# Ideal locations -> check constraint on distance

ampl.read('/home/luisa/Documents/FOR project/minimart.mod')

ampl.read_data('/home/luisa/Documents/FOR project/minimart-I-50.dat')

ampl.solve()

nRaw = ampl.getParameter('n')

n = nRaw.getValues().toList()

capacity = ampl.getParameter('capacity').getValues().toList()

# print(n)

distancesRaw = ampl.getParameter('sp')

distanceValues = distancesRaw.getValues().toList()

# print(distanceValues)

openingCost = ampl.getObjective('obj').getValues().toList()

buildingDataRaw = ampl.getVariable('built')

buildingCxRaw = ampl.getParameter('Cx')

buildingCyRaw = ampl.getParameter('Cy')

fcCostRaw = ampl.getParameter('Fc')

vcCostRaw = ampl.getParameter('Vc')

fcCost = fcCostRaw.getValues().toList()

vcCost = vcCostRaw.getValues().toList()

buildingCx = buildingCxRaw.getValues().toList()

buildingCy = buildingCyRaw.getValues().toList()

buildingData = buildingDataRaw.getValues().toList()

# setup coordinates

origVertices = []
xCoord = []
yCoord = []

for j in range(len(buildingData)):
    if buildingData[j][1] == 1:
        origVertices.append(j + 1)
        xCoord.append(buildingCxRaw[j + 1])
        yCoord.append(buildingCyRaw[j + 1])


# for k in range(len(origVertices)):
#     print("node " + str(origVertices[k]) + " has coordinates " + str(xCoord[k]) + ", " + str(yCoord[k]))

def findIdx(n1):
    # find index in list of a certain vertex
    for k in range(len(origVertices)):
        if n1 == origVertices[k]:
            return k


def distance(n1: int, n2: int):
    # calculate distance
    i1 = findIdx(n1)
    i2 = findIdx(n2)
    return np.sqrt((xCoord[i1] - xCoord[i2]) ** 2 + (yCoord[i1] - yCoord[i2]) ** 2)


def bestRank(n1: int, n2: int, card: int):
    # rank all available vertices according to a certain function,
    # such that the best point is chosen
    rankN1 = []
    rankN2 = []
    rank = []
    i1 = findIdx(n1)
    i2 = findIdx(n2)
    for idx in range(len(origVertices)):
        if idx != i1:
            d = distance(n1, origVertices[idx])
            rankN1.append([origVertices[idx], d, 0])
    for idx in range(len(origVertices)):
        if idx != i2:
            d = distance(n2, origVertices[idx])
            rankN2.append([origVertices[idx], d, 0])
    # print(rankN1)
    # print(rankN2)
    rankN1.sort(key=lambda x: x[1])
    rankN2.sort(key=lambda x: x[1])
    for idx in range(len(rankN1)):
        rankN1[idx][2] = idx
        rankN2[idx][2] = idx
    for j in range(len(rankN1)):
        tmp1 = rankN1[j][0]
        for k in range(len(rankN2)):
            if tmp1 == rankN2[k][0]:
                score = rankN1[j][2] + rankN2[k][2]
                rank.append([tmp1, score])
    rank.sort(key=lambda x: x[1])
    res = []
    for k in range(card):
        res.append(rank[k][0])
    return res


def bestRank1(n1: int, n2: int, card: int):
    # ranking, different function
    n2 = 1
    rankN1 = []
    rankN2 = []
    rank = []
    i1 = findIdx(n1)
    i2 = findIdx(n2)
    for idx in range(len(origVertices)):
        if idx != i1:
            d = distance(n1, origVertices[idx])
            rankN1.append([origVertices[idx], d, 0])
    for idx in range(len(origVertices)):
        if idx != i2:
            d = distance(n2, origVertices[idx])
            rankN2.append([origVertices[idx], d, 0])
    # print(rankN1)
    # print(rankN2)
    rankN1.sort(key=lambda x: x[1])
    rankN2.sort(key=lambda x: x[1])
    for idx in range(len(rankN1)):
        rankN1[idx][2] = idx
        rankN2[idx][2] = idx
    for j in range(len(rankN1)):
        tmp1 = rankN1[j][0]
        for k in range(len(rankN2)):
            if tmp1 == rankN2[k][0]:
                score = rankN1[j][2] + rankN2[k][2]
                rank.append([tmp1, score])
    rank.sort(key=lambda x: x[1])
    res = []
    for k in range(card):
        res.append(rank[k][0])
    return res


def medRank(n1: int, n2: int, card: int):
    # ranking, different function
    rankN1 = []
    rankN2 = []
    rank = []
    i1 = findIdx(n1)
    i2 = findIdx(n2)
    for idx in range(len(origVertices)):
        if idx != i1 and idx != i2:
            d = distance(n1, origVertices[idx])
            rankN1.append([origVertices[idx], d, 0])
            # for idx in range(len(origVertices)):
            #     if idx != i2:
            d = distance(n2, origVertices[idx])
            rankN2.append([origVertices[idx], d, 0])
    # print(rankN1)
    # print(rankN2)
    # rankN1.sort(key = lambda x: x[1])
    # rankN2.sort(key = lambda x: x[1])
    # for idx in range(len(rankN1)):
    #     rankN1[idx][2] = idx
    #     rankN2[idx][2] = idx
    for j in range(len(rankN1)):
        idx = rankN1[j][0]
        # for k in range(len(rankN2)):
        #     if tmp1 == rankN2[k][0]:
        score = np.abs(rankN1[j][1] - rankN2[j][1])
        rank.append([idx, score])
    rank.sort(key=lambda x: x[1])
    print("sorted rank med")
    print(rank)
    res = []
    for k in range(card):
        res.append(rank[k][0])
    return res


def neighbors(n1: int, card: int):
    # find card (cardinality) closest vertices to n1
    rankDist = []
    for j in range(len(origVertices)):
        if n1 != origVertices[j]:
            d = distance(n1, origVertices[j])
            rankDist.append([origVertices[j], d])
    rankDist.sort(key=lambda x: x[1])
    res = []
    for k in range(card):
        res.append(rankDist[k][0])
    return res


def plotLocations():
    # plot building locations
    plt.scatter(xCoord, yCoord, s=15)
    plt.xlabel('x coord')
    plt.ylabel('y coord')
    plt.title('Building Locations')
    for label in range(len(origVertices)):
        plt.text(xCoord[label], yCoord[label], origVertices[label])
    plt.show()


def refCostInit(route: None):
    # calculate refurbishment costs
    cost = fcCost[0]
    cost += distance(1, route[0]) * vcCost[0]
    cost += distance(1, route[len(route) - 1]) * vcCost[0]
    for item in range(len(route) - 1):
        cost += distance(route[item], route[item + 1]) * vcCost[0]
    return cost


def findCouples(extremes: None):
    # given a set of points (closest points to start)
    # divide them into couples, such that these couples' members are closest to each other
    couples = []
    idx = 0
    while len(extremes) > 0:
        print(len(extremes))
        print(idx)
        ext1 = extremes[idx]
        extremes.remove(ext1)
        ext2 = 1
        # it1 = 0
        # while it1 < len(extremes):
        #     ext1 = extremes[it1]
        #     extremes.remove(ext1)
        #     d = 1000
        #     for it2 in range(len(extremes)):
        #         if distance(ext1, extremes[it2]) < d:
        #             d = distance(ext1, extremes[it2])
        #             tmpMin = extremes[it2]
        couples.append([ext1, ext2])
        #     extremePts.remove(tmpMin)
        #     it1+=1
    return couples


def checkVisit(visits: None, n):
    # check whether a node was previously visited
    for it in range(len(visits)):
        if visits[it] == n: return False
    return True


def plotRoutes(routes: None):
    # plot nodes and corresponding arcs in the route
    for r in range(len(routes)):
        x = []
        y = []
        for it in range(len(routes[r])):
            idx = findIdx(routes[r][it])
            x.append(xCoord[idx])
            y.append(yCoord[idx])
        idx = findIdx(routes[r][0])
        x.append(xCoord[idx])
        y.append(yCoord[idx])
        plt.plot(x, y)
        color = iter(cm.rainbow(np.linspace(0, 1, len(routes[r]))))
        for label in range(len(routes[r])):
            plt.text(x[label], y[label], origVertices[findIdx(routes[r][label])])
    plt.xlabel('x coord')
    plt.ylabel('y coord')
    plt.title('Building Locations')
    plt.show()


# initialization

numTrucks = np.ceil((len(origVertices) - 1) / capacity[0])

totRefinedCostsB = [10000]
routesRefinedB = []

extremePts = neighbors(1, int(numTrucks))

couples = findCouples(extremePts)

routes = []
visited = []
innRoute = []
tmpList = origVertices.copy()

for it1 in range(len(couples)):
    routes.append([])
    innRoute.append(2)
    routes[it1].append(couples[it1][0])
    tmpList.remove(couples[it1][0])
    # visited.append(couples[it1][0])

# tmpList contains all the available nodes

tmpList.remove(1)

innerNodes = capacity[0]

# parallel filling, adding best ranking vertex (fixed point: origin of the route = 1)

for addition in range(int(numTrucks)):
    while len(tmpList) > 0:
        for r in range(len(couples)):
            beg = couples[r][1]
            if innRoute[r] <= innerNodes and len(tmpList) > 0:
                curr = routes[r][len(routes[r]) - 1]
                inn = innRoute[r]
                c = 1
                # the added node is the best ranking one, if available
                res = bestRank(curr, beg, c)
                add = res[c - 1]
                flag = checkVisit(tmpList, add)
                while flag and c <= len(origVertices):
                    res = neighbors(curr, c)
                    add = res[c - 1]
                    flag = checkVisit(tmpList, add)
                    c += 1
                print(add)
                routes[r].append(add)
                tmpList.remove(add)
                innRoute[r] += 1

# check that all available nodes are added
if len(tmpList) > 0:
    for it in range(len(origVertices)):
        if not (checkVisit(tmpList, origVertices[it])):
            neigh = neighbors(origVertices[it], 1)
            for r in len(routes):
                for c in len(routes[r]):
                    if routes[r][c] == neigh:
                        routes[r].append(neigh)

rCost = 0

# calculate cost in this configuration
for it in range(len(routes)):
    rCost += refCostInit(routes[it])

# add node 1 to all routes
for it1 in range(len(couples)):
    routes[it1].append(1)

totRefinedCostsB.append(rCost + openingCost)
routesRefinedB.append(routes)

for r in range(len(routesRefinedB)):
    plotRoutes(routesRefinedB[r])

print("First tentative cost:")
print(totRefinedCostsB[1])


def optimizeRoute(route: None):
    # consider all nodes within a route, order them s.t. the overall cost is lowered
    currCost = 0
    for item in range(len(route) - 1):
        currCost += distance(route[item], route[item + 1]) * vcCost[0]
    currCost += distance(route[0], route[len(route) - 1]) * vcCost[0]
    orderedRoute = []
    for sn in range(len(route)):
        tmpList = route.copy()
        tmpList.remove(route[sn])
        tmpOrd = [route[sn]]
        curr = route[sn]
        while len(tmpList) > 0:
            tmpD = 10000
            for item in range(len(route)):
                if not (checkVisit(tmpList, route[item])):
                    d = distance(curr, route[item])
                    if d < tmpD:
                        tmpD = d
                        nextIt = route[item]
            tmpOrd.append(nextIt)
            tmpList.remove(nextIt)
            curr = nextIt
        tmpCost = 0
        for item in range(len(tmpOrd) - 1):
            tmpCost += distance(tmpOrd[item], tmpOrd[item + 1]) * vcCost[0]
        tmpCost += distance(tmpOrd[0], tmpOrd[len(tmpOrd) - 1]) * vcCost[0]
        if currCost < tmpCost:
            orderedRoute.append([1, currCost, route])
        else:
            orderedRoute.append([route[sn], tmpCost, tmpOrd])
        # take best configuration
        orderedRoute.sort(key=lambda x: x[1])
        return orderedRoute[0]


optRouteB = []

# optimize all routes
for rout in range(len(routesRefinedB[0])):
    optTmp = optimizeRoute(routesRefinedB[0][rout])
    optRouteB.append(optTmp)

# print(optRouteB)

bestRoute = []
tot = 0

# evaluate total cost and build final configuration
for rout in range(len(optRouteB)):
    bestRoute.append(optRouteB[rout][2])
    tot += optRouteB[rout][1] + fcCost[0]

print("Refurbishment Cost: " + str(tot))

tot += openingCost[0]

print(bestRoute)

print("total cost is: " + str(tot))

plotRoutes(bestRoute)
