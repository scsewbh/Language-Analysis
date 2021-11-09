from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

a = "Increased Death"
b = "Increased Death"

print(similar(a, b))

def checkAllSyntax(a):
    a = a.lower()
    values = [] #index+1 is line number
    filename = 'finding36279Terms.txt'
    with open(filename) as f:
        for line in f:
            values.append(similar(a, line.lower()))
    f.close()

    N = 10
    res = sorted(range(len(values)), key=lambda sub: values[sub])[-N:]
    res.reverse()
    print(res)
    for x in res:
        print(values[x], end=" ")

checkAllSyntax("Convalescence after radiotherapy (finding)")