from src.evaluate import corrected_paired_test

# Sem seleção (run 20261106_192505) vs Com seleção (run 20261106_195503)
scores_no_sel = [0.732, 0.701, 0.736, 0.736, 0.731]
scores_with_sel = [0.724, 0.704, 0.731, 0.732, 0.734]

# Em 5-fold CV: test_train_ratio = 1 / (5 - 1) = 0.25
res = corrected_paired_test(scores_no_sel, scores_with_sel, test_train_ratio=0.25)
print(res)

median = [
    0.718124269634348,
    0.7000803809754914,
    0.7266258189759767,
    0.725420104343606,
    0.7333277723853433,
]

random_forest = [
    0.7205961600174379,
    0.6994646323707838,
    0.7275236593059937,
    0.7243114535306964,
    0.7328955350643047,
]

result = corrected_paired_test(
    median,
    random_forest,
    test_train_ratio=0.25,
)

print(result)