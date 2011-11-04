"""
Module for translating numbers into their English-word equivalents.

Was originally written for an ITA software challenge.

:author: Robert David Grant <robert.david.grant@gmail.com>

:copyright:
    Copyright 2011 Robert Grant

    Licensed under the Apache License, Version 2.0 (the "License"); you
    may not use this file except in compliance with the License.  You
    may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    implied.  See the License for the specific language governing
    permissions and limitations under the License.
"""

import string
import math


ones_digit_names = {
        '0':'',
        '1':'one', 
        '2':'two', 
        '3':'three',
        '4':'four',
        '5':'five',
        '6':'six',
        '7':'seven',
        '8':'eight',
        '9':'nine',
        }

tens_digit_teens = {
        '10':'ten',
        '11':'eleven',
        '12':'twelve',
        '13':'thirteen',
        '14':'fourteen',
        '15':'fifteen',
        '16':'sixteen',
        '17':'seventeen',
        '18':'eighteen',
        '19':'nineteen',
        }

tens_digit_names = {
        '2':'twenty',
        '3':'thirty',
        '4':'forty',
        '5':'fifty',
        '6':'sixty',
        '7':'seventy',
        '8':'eighty',
        '9':'ninety',
        }

group_names = {
        0:('', 0),
        1:('thousand', 8),
        2:('million', 7),
        3:('billion', 7),
        4:('trillion', 8),
        }

group_cache = {}

def group_digits(number,group_length=3):
    """Split a string into groups of length digits."""
    number_length = len(number)
    ngroups = number_length / group_length
    remainder = number_length % group_length

    start = 0
    groups = []
    if remainder > 0:
        groups.append(number[start:remainder])
        start += remainder

    for x in xrange(ngroups):
        end = start + group_length
        groups.append(number[start:end])
        start += group_length

    return groups


def translate_group(group):
    """Translate a group of three numbers to its English equivalent, with all
    zeros reported as the empty string.  Caches results for speed."""

    length = len(group)

    def translate_one_digit(x):
        return ones_digit_names[x]

    def translate_two_digit(x):
        if int(x) < 20:
            if x[0] == '0':
                return translate_one_digit(x[1])
            else:
                return tens_digit_teens[x]
        else:
            if x[0] == '0':
                return translate_one_digit(x[1])
            else:
                return tens_digit_names[x[0]] + translate_one_digit(x[1])

    def translate_three_digit(x):
        if x[0] == '0':
            return translate_two_digit(x[1:])
        elif x[1] == '0' and x[2] == '0':
            return ones_digit_names[x[0]] + 'hundred'
        else:
            return ones_digit_names[x[0]] + 'hundredand' + translate_two_digit(x[1:])

    try:
        return group_cache[group][0]
    except:
        if length == 1:
            translation = translate_one_digit(group)
            group_cache[group] = (translation, len(translation))
        elif length == 2:
            translation = translate_two_digit(group)
            group_cache[group] = (translation, len(translation))
        elif length == 3:
            translation = translate_three_digit(group)
            group_cache[group] = (translation, len(translation))
        else:
            raise TypeError
        return group_cache[group][0] # the phrase, not the length

def group_length(group):
    try:
        return group_cache[group][1] # the length, not the phrase
    except:
        return(len(translate_group(group)))

def translate_number(number):
    """Translate a number to its English equivalent."""
    number = str(number)
    groups = group_digits(number)
    groups.reverse()
    translation = []
    for x in xrange(len(groups)):
        if groups[x] != '000':
            translation.append(translate_group(groups[x]) + group_names[x][0])

    translation.reverse()
    return ''.join(translation)

def word_length(number):
    """Just return the string length of a number written out in
    characters."""
    number = str(number)
    groups = group_digits(number)
    groups.reverse()
    length = 0
    for x in xrange(len(groups)):
        length += group_length(groups[x]) + group_names[x][1]
    return length

def find_char(index):
    """Find the xth character in the string of concatenated numbers."""
    print_threshold = 0
    print_interval = int(1e6)
    running_count = 0
    i = 0 
    while i <= index:
        i = i+1
        length = word_length(i)
        running_count += length
        if running_count > print_threshold:
            print str(print_threshold/int(1e6)) + 'M'
            print_threshold += print_interval
        if running_count > index:
            word = translate_number(i)
            index_in_word = int(index-(running_count-length))
            return word[index_in_word]

def count_chars(number):
    """Count the chars if the numbers from 1:`number` are written out in
    words.
    """
    running_sum = 0
    for num in range(number):
        running_sum += word_length(num)
    return running_sum
