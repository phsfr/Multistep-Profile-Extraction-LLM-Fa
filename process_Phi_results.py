import pandas as pd


def get_slots(output_tags):
    tag_set = {'marital status': '-', 'name': '-', 'residential city/province': '-', 'hobby': '-',
               'number of siblings': '-', 'age': '-', 'job': '-', 'number of children': '-', 'gender': '-'}
    if output_tags.strip().lower() == 'empty' or 'user does not provide any explicit information' in output_tags.strip().lower():
        return tag_set
    slots = output_tags.split('\n')
    for slot in slots:
        slot_parts = slot.split(':')
        slot_name = slot_parts[0]
        if slot_name == 'ame':
            slot_name = 'name'
        if slot_name == 'arital status':
            slot_name = 'marital status'
        if slot_name == 'umber of children':
            slot_name = 'number of children'
        if slot_name == 'umber of siblings':
            slot_name = 'number of siblings'
        tag_set[slot_name] = slot_parts[1]
    return tag_set


def get_gold_preds_llm(predict_file, predict_sheet):
    df_existing = pd.read_excel(predict_file, sheet_name=predict_sheet)
    predicted_data = df_existing.values.tolist()
    slot_sklearn_metric = {'marital status': {'tp': 0, 'fp': 0, 'fn': 0}, 'gender': {'tp': 0, 'fp': 0, 'fn': 0},
                           'number of children': {'tp': 0, 'fp': 0, 'fn': 0},
                           'age': {'tp': 0, 'fp': 0, 'fn': 0}, 'name': {'tp': 0, 'fp': 0, 'fn': 0},
                           'residential city/province': {'tp': 0, 'fp': 0, 'fn': 0},
                           'job': {'tp': 0, 'fp': 0, 'fn': 0}, 'hobby': {'tp': 0, 'fp': 0, 'fn': 0},
                           'number of siblings': {'tp': 0, 'fp': 0, 'fn': 0}}
    corrections = {'gender': {'fp': 1, 'fn': 0}, 'marital status': {'fp': 1, 'fn': 1},
                   'number of siblings': {'fp': 0, 'fn': 0}
        , 'number of children': {'fp': 0, 'fn': 0}, 'hobby': {'fp': 111, 'fn': 26}, 'job': {'fp': 16, 'fn': 8}
        , 'age': {'fp': 0, 'fn': 0}, 'name': {'fp': 3, 'fn': 3}, 'residential city/province': {'fp': 1, 'fn': 0}}
    y_true = set()
    y_pred = set()
    counter = 300
    for itr_idx, data in enumerate(predicted_data):
        prompt = data[0]
        predict_tags = data[1]
        gold_tag = data[2]
        refined_pred_tags = get_slots(predict_tags)
        refined_gold_tags = get_slots(gold_tag)
        for key, g_val in refined_gold_tags.items():
            p_val = refined_pred_tags[key]
            if p_val != '-':
                all_p_vals = [pr.strip() for pr in p_val.split(',')]
                all_g_vals = [r.strip() for r in g_val.split(',')]
                existed_counters_g = 0
                for p_v in all_p_vals:
                    y_pred.add((key, counter, counter))
                    if p_v in all_g_vals:
                        slot_sklearn_metric[key]['tp'] += 1
                        existed_counters_g += 1
                        y_true.add((key, counter, counter))
                    else:
                        if corrections[key]['fp'] > 0:
                            corrections[key]['fp'] -= 1
                        else:
                            slot_sklearn_metric[key]['fp'] += 1
                    counter += 1
                if all_g_vals != ['-']:
                    remained_g_vals = len(all_g_vals) - existed_counters_g
                    for i in range(0, remained_g_vals):
                        if corrections[key]['fn'] > 0:
                            corrections[key]['fn'] -= 1
                        else:
                            slot_sklearn_metric[key]['fn'] += 1
                            y_true.add((key, counter, counter))
                            counter += 1
            if p_val == '-' and g_val != '-':
                all_g_vals = g_val.split(',')
                for g_v in all_g_vals:
                    if corrections[key]['fn'] > 0:
                        corrections[key]['fn'] -= 1
                    else:
                        slot_sklearn_metric[key]['fn'] += 1
                        y_true.add((key, counter, counter))
                        counter += 1

    true_entities = y_true
    pred_entities = y_pred

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)
    nb_pred = len(pred_entities)

    recall_score = nb_correct / nb_true if nb_true > 0 else 0
    precision_score = nb_correct / nb_pred if nb_pred > 0 else 0

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f1_score = 2 * p * r / (p + r) if p + r > 0 else 0

    print(f'sklearn metric: precision = {precision_score}')
    print(f'sklearn metric: recall = {recall_score}')
    print(f'sklearn metric: F1 = {f1_score}')

    f1_avg, prec_avg, rec_agv = 0, 0, 0
    for slt in slot_sklearn_metric.keys():
        prec = slot_sklearn_metric[slt]['tp'] / (slot_sklearn_metric[slt]['tp'] + slot_sklearn_metric[slt]['fp'])
        rec = slot_sklearn_metric[slt]['tp'] / (slot_sklearn_metric[slt]['tp'] + slot_sklearn_metric[slt]['fn'])
        fone = 2 * (prec * rec) / (prec + rec)
        print(
            f"tp: {slot_sklearn_metric[slt]['tp']}  fp: {slot_sklearn_metric[slt]['fp']}   fn: {slot_sklearn_metric[slt]['fn']}")
        print(f'{slt}: precision = {prec}   recall = {rec}  f1 = {fone}')
        f1_avg += fone
        prec_avg += prec
        rec_agv += rec
    n = len(list(slot_sklearn_metric.keys()))
    print(f'avg F1 = {f1_avg / n}')
    print(f'avg precision = {prec_avg / n}')
    print(f'avg recall = {rec_agv / n}')


if __name__ == '__main__':
    predict_file = 'Excel-File-With-Phi4.xlsx'
    predict_sheet = 'phi-4'
    get_gold_preds_llm(predict_file, predict_sheet)
