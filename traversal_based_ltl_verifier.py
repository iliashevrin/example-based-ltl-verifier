#!/usr/bin/env python3
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
import itertools
from utils import get_words_from_conditions, check_acceptance





def generate_traces_by_traversal(candidate):

    aut = spot.translate(candidate, 'parity', 'sbacc', 'state-based', 'complete', 'colored', 'deterministic')
    positive1, negative1 = generate_traces(aut, max_visits=1)
    positive2, negative2 = generate_traces(aut, max_visits=2)

    positive = positive1 + positive2
    negative = negative1 + negative2

    return positive, negative


def generate_traces(aut, max_visits=1, max_generate=200):

    positive = []
    negative = []
    paths = []

    for edge in aut.out(aut.get_init_state_number()):
        paths.append(([edge], {edge.src:1}))

    all_vars = aut.ap()

    while paths:

        path, visited = paths.pop(0)
        state = path[-1].dst

        if state in visited and visited[state] == max_visits:

            index = [edge.src for edge in path].index(state)
            conditions = [spot.bdd_format_formula(aut.get_dict(), edge.cond) for edge in path]
            words = get_words_from_conditions(conditions, index)

            props = visited.copy()
            props[state] += 1
            props = str([props[edge.src] for edge in path] + [props[state]])

            if words:
                if aut.intersects(spot.parse_word(words[0])):
                    positive.extend([(word, props) for word in words])
                else:
                    negative.extend([(word, props) for word in words])

                if len(positive) + len(negative) >= 200:
                    break

        else:

            if state not in visited:
                visited[state] = 1
            else:
                visited[state] += 1

            for edge in aut.out(state):
                paths.append((path + [edge], visited.copy()))

    return positive, negative
