import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
import itertools


def simulate_user(formula, ground_truth):

    aut = spot.translate(formula, 'parity', 'sbacc', 'state-based', 'complete', 'colored', 'deterministic')
    positive, negative = generate_traces(aut)

    print("all positive")
    for pos in positive:
        print(pos)

    print("")

    print("all negative")
    for neg in negative:
        print(neg)

    print("")

    for pos in positive:
        intersects = check_acceptance(spot.translate(ground_truth), pos)
        if not intersects:
            print(f'{pos} rejected, the formula is not the desired one')
        else:
            print(f'{pos} accepted, keep going')

    for neg in negative:
        intersects = check_acceptance(spot.translate(ground_truth), neg)
        if intersects:
            print(f'{neg} accepted, the formula is not the desired one')
        else:
            print(f'{neg} rejected, keep going')



def check_acceptance(aut, trace_str):
    
    trace = spot.parse_word(trace_str)
    if not aut.intersects(trace):
        return False
    
    trace_aut = trace.as_automaton()
    
    if spot.contains(aut, trace_aut):
        return True
    
    return None


def build_word(conditions, index):
    prefix = '; '.join(conditions[:index])
    cycle = "cycle{{ {} }}".format('; '.join(conditions[index:]))
    return cycle if not prefix else "{}; {}".format(prefix, cycle)


def separate(cond):
    return [str(formula) for formula in rec_separate(spot.formula(cond))]


def rec_separate(cond):
    
    if len(cond) < 2:
        return [cond]
    
    separate = [rec_separate(c) for c in cond]
    
    if cond._is(spot.op_Or):
        return [elem for c in separate for elem in c]
    
    if cond._is(spot.op_And):
        return [spot.formula.And(single) for single in itertools.product(*separate)]


def generate_traces(aut):

    positive = []
    negative = []
    paths = []

    for edge in aut.out(aut.get_init_state_number()):
        paths.append([edge])

    all_vars = aut.ap()

    while paths:

        path = paths.pop(0)
        state = path[-1].dst
        visited = [edge.src for edge in path]

        if state in visited:

            index = visited.index(state)
            conditions = [spot.bdd_format_formula(aut.get_dict(), edge.cond) for edge in path]
            
            words = []
            
            for single_cond in itertools.product(*[separate(cond) for cond in conditions]):

                word = build_word(single_cond, index)
                words.append(word)

            if words:
                if aut.intersects(spot.parse_word(words[0])):
                    positive.extend(words)
                else:
                    negative.extend(words)

        else:
            for edge in aut.out(state):
                paths.append(path + [edge])

    return positive, negative


if __name__ == "__main__":



    # sys.argv[1] = formula output of an LLM
    # sys.argv[2] = what we really want, i.e., ground truth
    simulate_user(sys.argv[1], sys.argv[2])