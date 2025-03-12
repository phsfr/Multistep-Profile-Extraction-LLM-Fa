import re
from hazm import word_tokenize
from persian_tools import digits
import jdatetime
from owlready2 import default_world

ontology_ns = "http://www.semanticweb.org/asus/ontologies/2024/2/expo_perinfex_onto_inferred_V2#"
UNK = 'نامشخص'


def find_gender(slot_value, stemmer, input_utter):
    yek = {'یه', 'یک'}
    male_form = ['آقا', 'اقا', 'مرد', 'پسر', ]
    female_form = ['خانم', 'خانوم', 'زن', 'دختر', 'مونث']
    male_pattern = f"({'|'.join(yek)} )?({'|'.join(female_form)}) نیستم\\.?"
    female_pattern = f"({'|'.join(yek)} )?({'|'.join(male_form)}) نیستم\\.?"

    gender = UNK
    if slot_value in male_form or re.match(male_pattern, slot_value):
        return 'Male'
    if slot_value in female_form or re.match(female_pattern, slot_value):
        return 'Female'

    words = word_tokenize(slot_value)
    stems = [stemmer.stem(word) for word in words]
    for stem in stems:
        if stem in male_form:
            return 'Male'
        if stem in female_form:
            return 'Female'

    return gender


def find_marital_stat(slot_value, lemmatizer, input_utter):
    def detect_neg_verb(lemmas, words):
        for idx, lem in enumerate(lemmas):
            if lem in verbs_lemm:
                verb = words[idx]
                if verb.startswith('ن') and (not lem.startswith('ن')):
                    return True
        return False

    verbs_lemm = ['کرد#کن', 'گرفت#گیر', 'داشت#دار', '#هست', 'شد#شو']
    mari_words = ['شوهر', 'زن', 'همسر', 'نامزد', 'داماد', 'عروس', 'متاهل', 'متأهل', 'تأهل', 'تاهل', 'ازدواج']
    singl_words = ['مجرد', 'تجرد', 'تنها']

    words = word_tokenize(slot_value)
    lemma = [lemmatizer.lemmatize(word) for word in words]
    marital_stat = {-1: UNK, 0: 'Single', 1: 'Married'}
    text = ' '.join(lemma)

    marri_pattern = f"({'|'.join(mari_words)})((\\s)({'|'.join(verbs_lemm)})(\\s)?)?\\.?"
    single_pattern = f"({'|'.join(singl_words)})((\\s)({'|'.join(verbs_lemm)})(\\s)?)?\\.?"
    match_marri = re.match(marri_pattern, text)
    match_single = re.match(single_pattern, text)
    marital_value = -1
    if match_marri:
        marital_value = 1
    elif match_single:
        marital_value = 0

    neg = detect_neg_verb(lemma, words)
    if neg and marital_value != -1:
        marital_value = 1 - marital_value
    return marital_stat[marital_value]


def find_num_siblings(slot_value, input_utter):
    num_sibls = {'brothers': -1, 'sisters': -1, 'siblings': -1}
    slot_value = slot_value.replace('\u200c', ' ').replace('یه', 'یک').replace('تک', 'یک').replace('دوتا', 'دو تا') \
        .replace('چهارتا', 'چهار تا').replace('تنها', 'یک نفر')
    if slot_value == 'برادر' or slot_value == 'خواهر' or slot_value == 'خواهربرادر':
        return num_sibls
    words = word_tokenize(slot_value)
    words_d = []
    for word in words:
        conv_d = digits.convert_from_word(word)
        if conv_d > 0:
            words_d.append(str(conv_d))
        else:
            words_d.append(word)
    conv_itm = ' '.join(words_d)
    brother_pattern = r"(\d+) ((دونه|تا|دانه|نفر)\s)?(\s?از\s?)?(داداشی|داداش|برادر)(\s)?\.?"
    brother_first_pattern = r"(داداشی|داداش|برادر) (\d+)(\s(دونه|تا|دانه|نفر))?(\s)?\.?"
    sister_pattern = r"(\d+) ((دونه|تا|دانه|نفر)\s)?(\s?از\s?)?(آبجی|ابجی|خواهر)(\s)?\.?"
    sister_first_pattern = r"(خواهر|آبجی|ابجی) (\d+)(\s(دونه|تا|دانه|نفر))?(\s)?\.?"
    both_pattern = r"(\d+) ((((دونه|تا|دانه|نفر)\s)?(خواهر(\s)?(و)?(\s)?برادر|برادر(\s)?(و)?(\s)?خواهر|نفر|فرزند|بچه))|(تاییم))(\s)?\.?"
    both_first_pattern = r"(خواهر\s?و?\s?برادر|برادر\s?و?\s?خواهر|آبجی\s?و?\s?داداش|داداش\s?و?\s?آبجی|فرزند|بچه) (\d+)(\s?(دونه|تا|دانه|نفر))?(\s)?\.?"

    no_sibl_pattern = r"((هیچ(\s))(((خواهر|ابجی|آبجی)(ی)?(?:(\s)?(یا|و)(\s)?)?(دادش|برادر)(ی)?)|((داداش|برادر)(ی)?(?:(\s)?(یا|و)(\s)?)?(آبجی|ابجی|خواهر)(ی)?))((\s)?ندارم)?)|((هیچ(\s))?(((خواهر|ابجی|آبجی)(ی)?(?:(\s)?(یا|و)(\s)?)?(دادش|برادر)(ی)?)|((داداش|برادر)(ی)?(?:(\s)?(یا|و)(\s)?)?(آبجی|ابجی|خواهر)(ی)?))((\s)?ندارم))(\s)?\.?"
    no_sister_pattern = "((هیچ|بدون) (خواهر|آبجی|ابجی)(ی)?((\s)?ندارم)?(\s)?\.?)|((خواهر|آبجی|ابجی)(ی)? ندارم(\s)?\.?)"
    no_brother_pattern = "((هیچ|بدون) (برادر|داداش)(ی)?((\s)?ندارم)?(\s)?\.?)|((داداش|برادر)(ی)?((\s)(هم|که))? ندارم(\s)?\.?)"

    just_sister_singl_pattern = r"خواهر|آبجی|ابجی"
    just_brother_singl_pattern = r"داداش|برادر"
    just_sister_plu_pattern = r"خواهرا|خواهر\s?ها|آبجی\s?ها|ابجی\s?ها"
    just_brother_plu_pattern = r"داداش(ی)?\s?ها|برادرا|برادر\s?ها|داداشا"
    just_both_pattern = r"(برادر(\s?و\s?)?\s?خواهر)|(خواهر(\s?و\s?)?\s?برادر)|(آبجی(\s?و\s?)?\s?داداش)|(داداش(\s?و\s?)?\s?آبجی)"
    match_sets = [re.match(brother_pattern, conv_itm),
                  re.match(sister_pattern, conv_itm),
                  re.match(both_pattern, conv_itm),
                  re.match(no_sibl_pattern, conv_itm),
                  re.match(no_sister_pattern, conv_itm),
                  re.match(no_brother_pattern, conv_itm),
                  re.match(brother_first_pattern, conv_itm),
                  re.match(sister_first_pattern, conv_itm),
                  re.match(both_first_pattern, conv_itm),
                  re.match(just_brother_singl_pattern, conv_itm),
                  re.match(just_sister_singl_pattern, conv_itm),
                  re.match(just_brother_plu_pattern, conv_itm),
                  re.match(just_sister_plu_pattern, conv_itm),
                  re.match(just_both_pattern, conv_itm), ]
    match_sizes = [len(match_set.group()) if match_set else 0 for match_set in match_sets]
    max_elem = max(match_sizes)
    max_idx = match_sizes.index(max_elem) if max_elem > 0 else -1

    if max_idx == 0 or max_idx == 6:
        group_idx = 1
        if max_idx == 6:
            group_idx = 2
        num_sibls['brothers'] = int(match_sets[max_idx].group(group_idx))
    elif max_idx == 1 or max_idx == 7:
        group_idx = 1
        if max_idx == 7:
            group_idx = 2
        num_sibls['sisters'] = int(match_sets[max_idx].group(group_idx))
    elif max_idx == 2 or max_idx == 8:
        group_idx = 1
        if max_idx == 8:
            group_idx = 2
        num_sibls['siblings'] = int(match_sets[max_idx].group(group_idx))
        if num_sibls['siblings'] <= 1:
            num_sibls['siblings'] = 0
            num_sibls['sisters'] = 0
            num_sibls['brothers'] = 0
    elif max_idx == 3:
        num_sibls['siblings'] = 0
        num_sibls['sisters'] = 0
        num_sibls['brothers'] = 0
    elif max_idx == 4:
        num_sibls['sisters'] = 0
    elif max_idx == 5:
        num_sibls['brothers'] = 0
    elif max_idx == 9:
        num_sibls['brothers'] = '1+'
    elif max_idx == 10:
        num_sibls['sisters'] = '1+'
    elif max_idx == 11:
        num_sibls['brothers'] = '2+'
    elif max_idx == 12:
        num_sibls['sisters'] = '2+'
    elif max_idx == 13:
        num_sibls['sisters'] = '1+'
        num_sibls['brothers'] = '1+'
        num_sibls['siblings'] = '2+'
    return num_sibls


def find_num_children(slot_value, input_utter):
    num_child = {'boys': -1, 'girls': -1, 'children': -1}
    slot_value = slot_value.replace('\u200c', ' ').replace('یه', 'یک').replace('تک', 'یک').replace('دوتا', 'دو تا') \
        .replace('چهارتا', 'چهار تا').replace('تنها', 'یک نفر').replace('یکی', 'یک')
    if slot_value == 'پسر' or slot_value == 'دختر' or slot_value == 'فرزند':
        return num_child
    words = word_tokenize(slot_value)
    words_d = []
    for word in words:
        conv_d = digits.convert_from_word(word)
        if conv_d > 0:
            words_d.append(str(conv_d))
        else:
            words_d.append(word)
    conv_itm = ' '.join(words_d)
    brother_pattern = r"(\d+) ((دونه|تا|دانه|نفر)\s)?(\s?از\s?)?((گل|آقا|کاکل)\s?)?(پسر)(\s)?\.?"
    brother_first_pattern = r"(پسر)(\s?(گل|آقا|دسته گل))? (\d+)(\s(دونه|تا|دانه|نفر))?(\s)?\.?"
    sister_pattern = r"(\d+) ((دونه|تا|دانه|نفر)\s)?(\s?از\s?)?(گل\s?)?(دختر)(\s)?\.?"
    sister_first_pattern = r"(دختر)(\s?(گل|آقا|دسته گل))? (\d+)(\s(دونه|تا|دانه|نفر))?(\s)?\.?"
    both_pattern = r"(\d+) ((دونه|تا|دانه|نفر)\s)?(دختر(\s)?(و)?(\s)?پسر|پسر(\s)?(و)?(\s)?دختر|خردسال|فرزند|بچه|نازدانه)(\s)?\.?"
    both_first_pattern = r"(دختر\s?و?\s?پسر|پسر\s?و?\s?دختر|فرزند|بچه|خردسال) (\d+)(\s?(دونه|تا|دانه|نفر))?(\s)?\.?"

    no_sibl_pattern = r"((هیچ(\s))(((دختر)(ی)?(?:(\s)?(یا|و)(\s)?)?(پسر)(ی)?)|((پسر)(ی)?(?:(\s)?(یا|و)(\s)?)?(دختر)(ی)?)|(فرزند(ی)?|فرزندانی|خردسال(ی)?|بچه(ای)?|بچه هایی))((\s)?ندارم)?)|((هیچ(\s))?(((دختر)(ی)?(?:(\s)?(یا|و)(\s)?)?(پسر)(ی)?)|(فرزند(ی)?|فرزندانی|خردسال(ی)?|بچه(ای)?|بچه هایی))((\s)?ندارم))(\s)?\.?"
    no_sister_pattern = "((هیچ|بدون) (دختر)(ی)?((\s)?ندارم)?(\s)?\.?)|((دختر)(ی)? ندارم(\s)?\.?)"
    no_brother_pattern = "((هیچ|بدون) (پسر)(ی)?((\s)?ندارم)?(\s)?\.?)|((پسر)(ی)?((\s)(هم|که))? ندارم(\s)?\.?)"

    just_sister_singl_pattern = r"دختر"
    just_brother_singl_pattern = r"پسر"
    just_sister_plu_pattern = r"دخترا|دختر\s?ها"
    just_brother_plu_pattern = r"پسرا|پسر\s?ها"
    just_both_pattern = r"(پسر(\s?و\s?)?\s?دختر)|(دختر(\s?و\s?)?\s?پسر)|فرزند|بچه|خردسال"

    multiple_both_pattern = r'(\d+) (?:(?:تا|دونه|دانه|عدد)\s?)?(\d+) قلو'
    match_sets = [re.match(brother_pattern, conv_itm),
                  re.match(sister_pattern, conv_itm),
                  re.match(both_pattern, conv_itm),
                  re.match(no_sibl_pattern, conv_itm),
                  re.match(no_sister_pattern, conv_itm),
                  re.match(no_brother_pattern, conv_itm),
                  re.match(brother_first_pattern, conv_itm),
                  re.match(sister_first_pattern, conv_itm),
                  re.match(both_first_pattern, conv_itm),
                  re.match(just_brother_singl_pattern, conv_itm),
                  re.match(just_sister_singl_pattern, conv_itm),
                  re.match(just_brother_plu_pattern, conv_itm),
                  re.match(just_sister_plu_pattern, conv_itm),
                  re.match(just_both_pattern, conv_itm),
                  re.match(multiple_both_pattern, conv_itm), ]
    match_sizes = [len(match_set.group()) if match_set else 0 for match_set in match_sets]
    max_elem = max(match_sizes)
    max_idx = match_sizes.index(max_elem) if max_elem > 0 else -1

    if max_idx == 0 or max_idx == 6:
        group_idx = 1
        if max_idx == 6:
            group_idx = 2
        num_child['boys'] = int(match_sets[max_idx].group(group_idx))
    elif max_idx == 1 or max_idx == 7:
        group_idx = 1
        if max_idx == 7:
            group_idx = 2
        num_child['girls'] = int(match_sets[max_idx].group(group_idx))
    elif max_idx == 2 or max_idx == 8:
        group_idx = 1
        if max_idx == 8:
            group_idx = 2
        num_child['children'] = int(match_sets[max_idx].group(group_idx))
    elif max_idx == 3:
        num_child['children'] = 0
        num_child['girls'] = 0
        num_child['boys'] = 0
    elif max_idx == 4:
        num_child['girls'] = 0
    elif max_idx == 5:
        num_child['boys'] = 0
    elif max_idx == 9:
        num_child['boys'] = '1+'
    elif max_idx == 10:
        num_child['girls'] = '1+'
    elif max_idx == 11:
        num_child['boys'] = '2+'
    elif max_idx == 12:
        num_child['girls'] = '2+'
    elif max_idx == 13:
        if ('پسر' in slot_value) and ('دختر' in slot_value):
            num_child['girls'] = '1+'
            num_child['boys'] = '1+'
        num_child['children'] = '2+'
    elif max_idx == 14:

        d1 = int(match_sets[max_idx].group(1))
        d2 = int(match_sets[max_idx].group(2))
        num_child['children'] = d1 * d2
    return num_child


def find_age(slot_value, input_utter):
    slot_value = slot_value.replace('\u200c', ' ')
    age_dict = digits.convert_from_word(slot_value)
    len_age = len(str(age_dict))
    if 'متولد' in slot_value or len_age == 4:
        if len_age == 2:
            age_dict = digits.convert_from_word('13' + str(age_dict))
        return jdatetime.datetime.now().year - age_dict
    if age_dict > 0:
        return age_dict
    if slot_value == 'نوجوان' or 'تینیجر' in slot_value or 'تین ایجر' in slot_value:
        return '15<,<18'
    if slot_value == 'جوان':
        return '18<,<40'
    if slot_value == 'میان سال':
        return '40<,<60'
    if slot_value == 'پیر' or slot_value == 'سالمند':
        return '<60'
    return -1


def find_residence(slot_value, onto_ns, input_utter):
    def find_residence_onto(itm):
        residence = {'Country': '', 'Region': '', 'City': '', 'Province': '', 'Section': ''}
        # print(itm)
        sparql_query = f"""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX perinfex:<{ontology_ns}>
                SELECT ?resist WHERE {{
                    ?resist perinfex:triggerWord "{itm}" .
                }}
            """
        results = list(default_world.sparql(sparql_query))
        # print(results)
        for result in results:
            entity = result
            if 'city' in str(entity):
                residence['City'] = result[0].label[0]
                province = result[0].hasProvince
                country = result[0].cityIsInCountry
                if province is not None:
                    region = province.belongsToRegion
                    if region is not None:
                        residence['Region'] = region.label[0]
                residence['Country'] = country.label[0]
                if province is not None:
                    residence['Province'] = province.label[0]
                return True, residence, 'city'
            if 'province' in str(entity):
                residence['Province'] = result[0].label[0]
                country = result[0].provinceIsInCountry
                region = result[0].belongsToRegion
                if region is not None:
                    residence['Region'] = region.label[0]
                residence['Country'] = country.label[0]
                return True, residence, 'province'
            if 'Region' in str(entity):
                residence['Region'] = result[0].label[0]
                residence['Country'] = onto_ns.Country("country_ایران").label[0]
                return True, residence, 'region'
            if 'country' in str(entity):
                residence['Country'] = result[0].label[0]
                return True, residence, 'country'
        return False, residence, ''

    itm = slot_value.replace('.', '').replace('،', '').replace(',', '').replace('؛', '')
    org_itm = itm
    itm = itm.replace(' ', '').replace('\u200c', '')
    found, residence, itm_type = find_residence_onto(itm)
    if not found and itm.endswith('ی'):
        found, residence, itm_type = find_residence_onto(itm[:-1])
    if not found and itm.endswith('یم'):
        found, residence, itm_type = find_residence_onto(itm[:-2])
    if not found and itm.endswith('یام'):
        found, residence, itm_type = find_residence_onto(itm[:-3])
    if not found and itm.endswith('ایم'):
        found, residence, itm_type = find_residence_onto(itm[:-3])
    if not found and itm.endswith('ام'):
        found, residence, itm_type = find_residence_onto(itm[:-2])
    if not found and itm.endswith('م'):
        found, residence, itm_type = find_residence_onto(itm[:-1])
    if not found:
        residence = {'Country': '', 'Region': '', 'City': '', 'Province': '', 'Section': org_itm}
        itm_type = 'section'
    return residence, itm_type


def find_name_gender(slot_value, input_utter):
    def find_name_onto(itm):
        name_gender = {'Name': '', 'Gender': ''}
        sparql_query = f"""
                            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                            PREFIX perinfex:<{ontology_ns}>
                            SELECT ?name WHERE {{
                                ?name rdf:type perinfex:FirstName .
                                ?name perinfex:triggerWord "{itm}" .
                            }}
                        """
        results = list(default_world.sparql(sparql_query))
        for result in results:
            entity = result
            name_gender['Name'] = result[0].label[0]
            if result[0].firstNameGender:
                name_gender['Gender'] = result[0].firstNameGender[0].label[0]
                return True, name_gender
        return False, name_gender

    slot_value = slot_value.replace('.', '').replace('،', '').replace(',', '').replace('؛', '')
    org_itm = slot_value
    slot_value = slot_value.replace(' ', '').replace('\u200c', '')
    found, name_gender = find_name_onto(slot_value)
    if not found and slot_value.endswith('م'):
        found, name_gender = find_name_onto(slot_value[:-1])
    if not found and slot_value.endswith('ه'):
        found, name_gender = find_name_onto(slot_value[:-1])
    if not found and slot_value.endswith('ست'):
        found, name_gender = find_name_onto(slot_value[:-2])
    if not found and len(org_itm.split()) > 1:
        itm = org_itm.split()[0].replace('\u200c', '')
        found, name_gender = find_name_onto(itm)
    if not found:
        name_gender = {'Name': org_itm, 'Gender': ''}
    return name_gender


def find_job(slot_value, onto_ns, lemmatizer):
    def find_job_onto(itm):
        sparql_query = f"""
                            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                            PREFIX perinfex:<{ontology_ns}>
                            SELECT ?job WHERE {{
                                ?job perinfex:triggerWord "{itm}" .
                                ?job rdf:type ?type .
                                ?type rdfs:subClassOf* perinfex:Job .
                            }}
                        """
        results = list(default_world.sparql(sparql_query))
        for result in results:
            return result[0]
        return None

    jobless_expr = {'کارنمیکنم', 'کارندارم', 'بیکار', 'بیکارم', 'بیکارهستم'}
    slot_value = slot_value.replace('.', '').replace('،', '').replace(',', '').replace('؛', '')
    job = lemmatizer.lemmatize(slot_value)
    slot_value = slot_value.replace(' ', '').replace('\u200c', '')
    if slot_value in jobless_expr:
        return onto_ns.Job('job_' + 'بیکار')  # 'بیکار'
    onto_job = find_job_onto(slot_value)
    if onto_job is None and slot_value.endswith('ه'):
        onto_job = find_job_onto(slot_value[:-1])
    if onto_job is None and slot_value.endswith('م'):
        onto_job = find_job_onto(slot_value[:-1])
    if onto_job is not None:
        return onto_job
    onto_job = onto_ns.Job('job_' + job.replace(' ', '_'))
    onto_job.label.append(job)
    onto_job.triggerWord.append(slot_value)
    return onto_job


def find_hobby(slot_value, onto_ns):
    slot_value = slot_value.replace('.', '').replace('،', '').replace(',', '').replace('؛', '')
    trigger = slot_value.replace(' ', '').replace('\u200c', '')
    sparql_query = f"""
                        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        PREFIX perinfex:<{ontology_ns}>
                        SELECT ?hobby WHERE {{
                            ?hobby perinfex:triggerWord "{trigger}" .
                            ?hobby rdf:type perinfex:Hobby .
                        }}
                    """
    results = list(default_world.sparql(sparql_query))
    for result in results:
        return result[0]

    onto_hobby = onto_ns.Hobby('hobby_' + trigger)
    onto_hobby.label.append(slot_value)
    onto_hobby.triggerWord.append(trigger)
    return onto_hobby