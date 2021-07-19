# -*- coding: utf-8 -*-

#   Foma: a finite-state toolkit and library.                                 #
#   Copyright © 2008-2015 Mans Hulden                                         #

#   This file is part of foma.                                                #

#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at                                   #

#      http://www.apache.org/licenses/LICENSE-2.0                             #

#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #

from sys import maxsize, version_info
from ctypes import *
from ctypes.util import find_library

fomalibpath = find_library('foma')

def find_library_trying_harder(executable_name):
    """
    If `foo` comes from homebrew, find_library might not find `libfoo`.
    But if `foo` is on `$PATH`, look for `libfoo` nearby.

    For example, if `foo` is at `/bar/bin/foo`, check if
    `/bar/lib/libfoo.dylib` exists.
    """
    import distutils.spawn
    import os.path

    executable = distutils.spawn.find_executable(executable_name)
    if executable is None:
        return

    for libdir_name in ['lib', 'lib64']:
        libdir = os.path.join(
            os.path.dirname(os.path.realpath(executable)),
            '..',
            libdir_name)
        for prefix in ['', 'lib']:
            for suffix in ['.so', '.dylib']:
                filename = prefix + executable_name + suffix
                path = os.path.join(libdir, filename)
                if os.path.isfile(path):
                    return path

if fomalibpath is None:
    fomalibpath = find_library_trying_harder('foma')
if fomalibpath is None:
    raise Exception("libfoma was not found")

foma = cdll.LoadLibrary(fomalibpath)

class FSTstruct(Structure):
    _fields_ = [
        ("name", c_char * 40),
        ("arity", c_int),
        ("arccount", c_int),
        ("statecount", c_int),
        ("linecount", c_int),
        ("finalcount", c_int),
        ("pathcount", c_longlong),
        ("is_deterministic", c_int),
        ("is_pruned", c_int),
        ("is_minimized", c_int),
        ("is_epsilon_free", c_int),
        ("is_loop_free", c_int),
        ("is_completed", c_int),
        ("arcs_sorted_in", c_int),
        ("arcs_sorted_out", c_int),
        ("fsm_state", c_void_p),
        ("sigma", c_void_p),
        ("medlookup", c_void_p)
    ]


foma_fsm_parse_regex = foma.fsm_parse_regex
foma_fsm_parse_regex.restype = POINTER(FSTstruct)

foma_fsm_create_letter_lookup = foma.fsm_create_letter_lookup
foma_fsm_create_letter_lookup.restype = c_void_p

foma_apply_init = foma.apply_init
foma_apply_init.restype = c_void_p
foma_apply_med_init = foma.apply_med_init
foma_apply_med_init.restype = c_void_p
foma_apply_clear = foma.apply_clear
foma_apply_words = foma.apply_words
foma_apply_words.restype = c_char_p
foma_apply_lower_words = foma.apply_lower_words
foma_apply_lower_words.restype = c_char_p
foma_apply_upper_words = foma.apply_upper_words
foma_apply_upper_words.restype = c_char_p
foma_apply_down = foma.apply_down
foma_apply_down.restype = c_char_p
foma_apply_up = foma.apply_up
foma_apply_up.restype = c_char_p

foma_apply_med_set_med_limit = foma.apply_med_set_med_limit
foma_apply_med_set_med_cutoff = foma.apply_med_set_med_cutoff
foma_apply_med_get_cost = foma.apply_med_get_cost
foma_apply_med_get_cost.restype = c_int
foma_apply_med_set_heap_max = foma.apply_med_set_heap_max

foma_apply_med = foma.apply_med
foma_apply_med.restype = c_char_p

foma_apply_set_space_symbol = foma.apply_set_space_symbol
foma_fsm_count = foma.fsm_count
foma_fsm_topsort = foma.fsm_topsort
foma_fsm_topsort.restype = POINTER(FSTstruct)
foma_fsm_minimize = foma.fsm_minimize
foma_fsm_minimize.restype = POINTER(FSTstruct)
foma_fsm_union = foma.fsm_union
foma_fsm_union.restype = POINTER(FSTstruct)
foma_fsm_intersect = foma.fsm_intersect
foma_fsm_intersect.restype = POINTER(FSTstruct)
foma_fsm_minus = foma.fsm_minus
foma_fsm_minus.restype = POINTER(FSTstruct)
foma_fsm_compose = foma.fsm_compose
foma_fsm_compose.restype = POINTER(FSTstruct)
foma_fsm_concat = foma.fsm_concat
foma_fsm_concat.restype = POINTER(FSTstruct)
foma_fsm_copy = foma.fsm_copy
foma_fsm_copy.restype = POINTER(FSTstruct)
foma_fsm_complement = foma.fsm_complement
foma_fsm_complement.restype = POINTER(FSTstruct)
foma_fsm_lower = foma.fsm_lower
foma_fsm_lower.restype = POINTER(FSTstruct)
foma_fsm_upper = foma.fsm_upper
foma_fsm_upper.restype = POINTER(FSTstruct)
foma_fsm_minimize = foma.fsm_minimize
foma_fsm_minimize.restype = POINTER(FSTstruct)
foma_fsm_destroy = foma.fsm_destroy
foma_fsm_equivalent = foma.fsm_equivalent
foma_fsm_equivalent.restype = c_int
foma_fsm_isempty = foma.fsm_isempty
foma_fsm_isempty.restype = c_int
foma_fsm_flatten = foma.fsm_flatten
foma_fsm_flatten.restype = POINTER(FSTstruct)
foma_apply_set_space_symbol = foma.apply_set_space_symbol
foma_fsm_read_binary_file = foma.fsm_read_binary_file
foma_fsm_read_binary_file.restype = POINTER(FSTstruct)
foma_fsm_get_library_version_string = foma.fsm_get_library_version_string
foma_fsm_get_library_version_string.restype = c_char_p

__version__ = foma.fsm_get_library_version_string().decode('UTF-8')

"""Define functions."""
foma_add_defined = foma.add_defined
foma_add_defined.restype = c_int

# The foma that homebrew install is the 0.9.18 release from 2015 at
# https://bitbucket.org/mhulden/foma/downloads/. It does not export
# `add_defined_function`. So we only expose it to python if it is
# available.
#
# $ curl -L https://bintray.com/homebrew/bottles/download_file?file_path=foma-0.9.18.big_sur.bottle.1.tar.gz \
#       | tar xf - --to-stdout foma/0.9.18/lib/libfoma.0.9.18.dylib \
#       | objdump -t /dev/stdin | grep add_defined
#
# 00000000000040b2 l     F __TEXT,__text _add_defined_function
# 0000000000004178 g     F __TEXT,__text _add_defined
if hasattr(foma, 'add_defined_function'):
    foma_add_defined_function = foma.add_defined_function
    foma_add_defined_function.restype = c_int

defined_networks_init = foma.defined_networks_init
defined_networks_init.restype = c_void_p
defined_functions_init = foma.defined_functions_init
defined_functions_init.restype = c_void_p

"""Trie functions."""
fsm_trie_init = foma.fsm_trie_init
fsm_trie_init.restype = c_void_p
fsm_trie_add_word = foma.fsm_trie_add_word
fsm_trie_done = foma.fsm_trie_done
fsm_trie_done.restype = POINTER(FSTstruct)


class FSTnetworkdefinitions(object):
    def __init__(self):
        self.defhandle = defined_networks_init(None)


class FSTfunctiondefinitions(object):
    def __init__(self):
        self.deffhandle = defined_functions_init(None)


class FST(object):
    networkdefinitions = FSTnetworkdefinitions()
    functiondefinitions = FSTfunctiondefinitions()

    # Generalize over Python2 and Python3 types
    string_type  = str   if version_info[0] > 2 else basestring
    text_type    = str   if version_info[0] > 2 else unicode
    binary_type  = bytes if version_info[0] > 2 else str

    @classmethod
    def define(cls, definition, name):
        """Defines an FSM constant; can be supplied regex or existing FSM."""
        name = cls.encode(name)
        if isinstance(definition, FST):
            retval = foma.add_defined(c_void_p(cls.networkdefinitions.defhandle), foma_fsm_copy(definition.fsthandle), c_char_p(name))
        elif isinstance(definition, FST.string_type):
            regex = cls.encode(definition)
            retval = foma.add_defined(c_void_p(cls.networkdefinitions.defhandle), foma_fsm_parse_regex(c_char_p(regex), c_void_p(cls.networkdefinitions.defhandle), c_void_p(cls.functiondefinitions.deffhandle)), c_char_p(name))
        else:
            raise ValueError("Expected str, unicode, or FSM")

    @classmethod
    def definef(cls, prototype, definition):
        """Defines an FSM function."""
        # Prototype is a 2-tuple (name, (arg1name, ..., argname))
        # Definition is regex using prototype variables
        name = cls.encode(prototype[0] + '(')
        if isinstance(definition, FST.string_type):
            numargs = len(prototype[1])
            for i in range(numargs):
                definition = definition.replace(prototype[1][i], "@ARGUMENT0%i@" % (i+1))
            regex = cls.encode(definition + ';')
            retval = foma.add_defined_function(c_void_p(cls.functiondefinitions.deffhandle), c_char_p(name), c_char_p(regex), c_int(numargs))
        else:
            raise ValueError("Expected regex as definition")
         
    @classmethod
    def wordlist(cls, wordlist, minimize = True):
        """Create FSM directly from wordlist.
           Returns a trie-shaped deterministic automaton if not minimized."""
        th = fsm_trie_init()
        for w in wordlist:
            thisword = cls.encode(w)
            fsm_trie_add_word(c_void_p(th), c_char_p(thisword))
        fsm = cls()
        fsm.fsthandle = fsm_trie_done(c_void_p(th))
        if minimize:
            fsm.fsthandle = foma_fsm_minimize(fsm.fsthandle)
        return fsm

    @classmethod
    def load(cls, filename):
        """Load binary FSM from file."""
        fsm = cls()
        fsm.fsthandle = foma_fsm_read_binary_file(c_char_p(FST.encode(filename)))
        if not fsm.fsthandle:
            raise ValueError("File error.")
        return fsm

    @staticmethod
    def encode(string):
        # type: (Any) -> FST.binary_type
        """Makes sure str and unicode are converted."""
        if isinstance(string, FST.text_type):
            return string.encode('utf8')
        elif isinstance(string, FST.binary_type):
            return string
        else:
            return FST.encode(str(string))

    @staticmethod
    def decode(text):
        if text is None:
            return None
        elif isinstance(text, FST.binary_type):
            # Assume output is UTF-8 encoded:
            return text.decode('UTF-8')
        else:
            assert isinstance(text, FST.text_type)
            return text

    def __init__(self, regex = False):
        if regex:
            self.regex = self.encode(regex)
            self.fsthandle = foma_fsm_parse_regex(c_char_p(self.regex), c_void_p(self.networkdefinitions.defhandle), c_void_p(self.functiondefinitions.deffhandle))
            if not self.fsthandle:
                raise ValueError("Syntax error in regex")
        else:
            self.fsthandle = None
        self.getitemapplyer = None

    def __getitem__(self, key):
        if not self.fsthandle:
            raise KeyError('FST not defined')
        if not self.getitemapplyer:
            self.getitemapplyer = foma_apply_init(self.fsthandle)
        result = []
        output = foma_apply_down(c_void_p(self.getitemapplyer), c_char_p(self.encode(key)))
        while True:
            if output == None:
                return result
            else:
                result.append(output)
                output = foma_apply_down(c_void_p(self.getitemapplyer), None)
            
    def __del__(self):
        if self.fsthandle:
            foma_fsm_destroy(self.fsthandle)

    def __str__(self):
        if not self.fsthandle:
            raise ValueError('FSM not defined')
        foma_fsm_count(self.fsthandle)
        s  = 'Name: %s\n' % self.fsthandle.contents.name
        s += 'States: %i\n' % self.fsthandle.contents.statecount
        s += 'Transitions: %i\n' % self.fsthandle.contents.arccount
        s += 'Final states: %i\n' % self.fsthandle.contents.finalcount
        s += 'Deterministic: %i\n' % self.fsthandle.contents.is_deterministic
        s += 'Minimized: %i\n' % self.fsthandle.contents.is_minimized
        s += 'Arity: %i\n' % self.fsthandle.contents.arity
        return s

    def __len__(self):
        if self.fsthandle:
            if self.fsthandle.contents.pathcount == -3: # UNKNOWN
                self.fsthandle = foma_fsm_topsort(self.fsthandle)
            if self.fsthandle.contents.pathcount == -1: # CYCLIC
                raise ValueError("FSM is cyclic")
            if self.fsthandle.contents.pathcount == -2: # OVERFLOW
                return maxsize
            return self.fsthandle.contents.pathcount
        else:
            raise ValueError("FSM not defined")
        
    def __add__(self, other):
        return self.concat(other)

    def __sub__(self, other):
        return self.minus(other)
    
    def __le__(self, other):
        if self.fsthandle and other.fsthandle:
            return bool(c_int(foma_fsm_isempty(foma_fsm_minimize(foma_fsm_minus(foma_fsm_copy(self.fsthandle),foma_fsm_copy(other.fsthandle))))))
        else:
            raise ValueError('Undefined FST')

    def __lt__(self, other):
        if self.fsthandle and other.fsthandle:
            return (not self.__eq__(other)) and bool(c_int(foma_fsm_isempty(foma_fsm_minimize(foma_fsm_minus(foma_fsm_copy(self.fsthandle),foma_fsm_copy(other.fsthandle))))))
        else:
            raise ValueError('Undefined FST')        
        
    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersect(other)

    def __eq__(self, other):
        if self.fsthandle and other.fsthandle:
            return bool(c_int(foma_fsm_equivalent(foma_fsm_copy(self.fsthandle), foma_fsm_copy(other.fsthandle))))
        else:
            raise ValueError('Undefined FST')
    
    def __ne__(self, other):
        return not(self.__eq__(other))

    def __contains__(self, word):
        af = self.apply_down(word)
        try:
            i = af.next()
            return True
        except StopIteration:
            return False
                
    def __call__(self, other):
        if isinstance(other, FST.string_type):
            return FST("{" + other + "}").compose(self)
        else:
            return other.compose(self)
    
    def __invert__(self):
        new = FST()
        new.fsthandle = self._fomacallunary(foma_fsm_complement)
        return new

    def __iter__(self):
        return self._apply(foma_apply_upper_words, word = None, tokenize = False)
    
    def _apply(self, applyf, word = None, tokenize = False):
        if not self.fsthandle:
            raise ValueError('FST not defined')
        applyerhandle = foma_apply_init(self.fsthandle)

        if tokenize:
            toksym = '\x07'
            foma_apply_set_space_symbol(c_void_p(applyerhandle), c_char_p(toksym))
        if word:
            output = applyf(c_void_p(applyerhandle), c_char_p(self.encode(word)))
        else:
            output = applyf(c_void_p(applyerhandle))
        while True:
            if output == None:
                foma_apply_clear(c_void_p(applyerhandle))
                return
            else:
                if tokenize:
                    yield output[:-1].split('\x07')
                else:
                    yield self.decode(output)
            if word:
                output = applyf(c_void_p(applyerhandle), None)
            else:
                output = applyf(c_void_p(applyerhandle))
                    
    def words(self, tokenize = False):
        return self._apply(foma_apply_words, word = None, tokenize = tokenize)

    def lowerwords(self, tokenize = False):
        return self._apply(foma_apply_lower_words, word = None, tokenize = tokenize)
                    
    def upperwords(self, tokenize = False):
        return self._apply(foma_apply_upper_words, word = None, tokenize = tokenize)
        
    def apply_down(self, word, tokenize = False):
        return self._apply(foma_apply_down, word = word, tokenize = tokenize)

    def apply_up(self, word, tokenize = False):
        if self.fsthandle:
            return self._apply(foma_apply_up, word = word, tokenize = tokenize)
        else:
            raise ValueError('Undefined FST')


    def apply_med(self, word):
        if not self.fsthandle:
            raise ValueError('FST not defined')
        
        medh = foma_apply_med_init(self.fsthandle)
        foma_apply_med_set_heap_max(c_void_p(medh),8388608+1)
        foma_apply_med_set_med_limit(c_void_p(medh), 10)
        foma_apply_med_set_med_cutoff(c_void_p(medh), 10)

        output = foma_apply_med(c_void_p(medh), c_char_p(self.encode(word)))

        while True:
            if output == None:
                return
            else:
                yield self.decode(output), foma_apply_med_get_cost(c_void_p(medh))

            output = foma_apply_med(c_void_p(medh), None)


    def _fomacallunary(self, func, minimize = True):
        if self.fsthandle:
            handle = func(foma_fsm_copy(self.fsthandle))
            if minimize:
                handle = foma_fsm_minimize(handle)
            return handle
        else:
            raise ValueError('Undefined FST')
        
    def _fomacallbinary(self, other, func, minimize = True):
        if self.fsthandle and other.fsthandle:
            handle = func(foma_fsm_copy(self.fsthandle), foma_fsm_copy(other.fsthandle))
            if minimize:
                handle = foma_fsm_minimize(handle)
            return handle
        else:
            raise ValueError('Undefined FST')
        
    def union(self, other, minimize = True):
        new = FST()
        new.fsthandle = self._fomacallbinary(other, foma_fsm_union, minimize)
        return new

    def intersect(self, other, minimize = True):
        new = FST()
        new.fsthandle = self._fomacallbinary(other, foma_fsm_intersect, minimize)
        return new

    def minus(self, other, minimize = True):
        new = FST()
        new.fsthandle = self._fomacallbinary(other, foma_fsm_minus, minimize)
        return new

    def concat(self, other, minimize = True):
        new = FST()
        new.fsthandle = self._fomacallbinary(other, foma_fsm_concat, minimize)
        return new

    def compose(self, other, minimize = True):
        new = FST()
        new.fsthandle = self._fomacallbinary(other, foma_fsm_compose, minimize)
        return new
    
    def lower(self, minimize = True):
        new = FST()
        new.fsthandle = self._fomacallunary(foma_fsm_lower, minimize)
        return new

    def upper(self, minimize = True):
        new = FST()
        new.fsthandle = self._fomacallunary(foma_fsm_upper, minimize)
        return new

    def flatten(self):
        new = FST()
        eps_sym = FST('□')
        new.fsthandle = foma_fsm_flatten(foma_fsm_copy(self.fsthandle), foma_fsm_copy(eps_sym.fsthandle))
        return new
    
class MTFSM(FST):

    def __init__(self, regex = False, numtapes = 2):
        if isinstance(regex, FST.string_type):
            FST.__init__(self, regex)
            eps_sym = FST('□')
            self.fsthandle = foma_fsm_flatten(foma_fsm_copy(self.fsthandle), foma_fsm_copy(eps_sym.fsthandle))
            self.numtapes = numtapes
        elif isinstance(regex, FST):
            self.fsthandle = foma_fsm_copy(regex.fsthandle)
            self.regex = None
            self.numtapes = numtapes
        else:
            self.fsthandle = None
            self.regex = None
        
        
    def __str__(self):
        if not self.fsthandle:
            raise ValueError('FSM not defined')
        foma_fsm_count(self.fsthandle)
        s  = 'Name: %s\n' % self.fsthandle.contents.name
        s += 'States: %i\n' % self.fsthandle.contents.statecount
        s += 'Transitions: %i\n' % self.fsthandle.contents.arccount
        s += 'Final states: %i\n' % self.fsthandle.contents.finalcount
        s += 'Deterministic: %i\n' % self.fsthandle.contents.is_deterministic
        s += 'Minimized: %i\n' % self.fsthandle.contents.is_minimized
        s += 'Numtapes: %i\n' % self.numtapes
        return s
    
    def generate(self, word):

        m = self.numtapes
        regx = (u'[{' + word + u'}/□ .o. [? 0:?^' + str(m-1) + ']*].l')
        reg = FST(regx)
        gr = FST()
        gr.fsthandle = foma_fsm_copy(self.fsthandle)
        res = MTFSM(reg.intersect(gr), numtapes = m)
        return res
    

    def parse(self, word):
        #[word/□ .o. [0:?^(numtapes-1) ?]*].l & Grammar ;
        m = self.numtapes
        regx = (u'[{' + word + u'}/□ .o. [0:?^' + str(m-1) + u' ?]*].l')
        reg = FST(regx)
        gr = FST()
        gr.fsthandle = foma_fsm_copy(self.fsthandle)
        res = MTFSM(reg.intersect(gr), numtapes = m)        
        return res
    
    def join(self, other):
        """Joins two multitape FSMs by composition. E.g.
            [ a d ]     [ c □ □ ]                  [ a d □ ]
        A = [ b e ] B = [ d f g ], and A.join(B) = [ b e □ ]
            [ c □ ]     [ e f g ]                  [ c □ □ ]
                                                   [ d f g ]
                                                   [ e f g ] """

        m = self.numtapes        
        n = other.numtapes
        pada = FST('[0:□^' + str(m) +' [0:?^' + str(n-1) + ' - 0:□^' + str(n-1) + '] | ?^' + str(m) + ' 0:?^' + str(n-1) + ']*')
        padb = FST('[[0:?^' + str(m-1) + ' - 0:□^' + str(m-1) + ' ] 0:□^' + str(n) + '| 0:?^' + str(m-1) + ' ?^' + str(n) + ']*')
        extenda = self.compose(pada).lower()
        extendb = other.compose(padb).lower()
        flt = FST('~[?^' + str(m+n-1) +'* [□^' + str(m) + ' ?^' + str(n-1) + ' [?^' + str(m-1) + ' □^' + str(n) + ' |[?^' + str(m-1) +' - □^' + str(m-1) + ' ] □ [?^' + str(n-1) + ' - □^' + str(n-1) + ' ]] | ?^' + str(m-1) + ' □^' + str(n) + ' [□^' + str(m) + ' ?^' + str(n-1) + ' |[?^' + str(m-1) + ' - □^' + str(m-1) + ' ] □ [?^' + str(n-1) + ' - □^' + str(n-1) + ' ]]] ?*]')
        res = extenda & extendb & flt
        result = MTFSM(res, m + n - 1)
        return result

    def _apply(self, applyf, word = None):
        if not self.fsthandle:
            raise ValueError('FST not defined')
        applyerhandle = foma_apply_init(self.fsthandle)
        output = applyf(c_void_p(applyerhandle))
        while True:
            if output == None:
                foma_apply_clear(c_void_p(applyerhandle))
                return
            else:
                yield output
            if word:
                output = applyf(c_void_p(applyerhandle), None)
            else:
                output = applyf(c_void_p(applyerhandle))

    def __iter__(self):
        return self._mtwords()

    def __add__(self, other):
        return self.join(other)

    def _fmt(self, word):
        cols = word
        colchunks = [map(lambda z: len(z), word[x:x+self.numtapes]) for x in range(0, len(word), self.numtapes)]
        col_widths = [max(x) for x in colchunks]
        format = '  '.join(['%%-%ds' % width for width in col_widths])
        # string to rows
        rows = [[word[y] for y in range(x, len(word), self.numtapes)] for x in range(self.numtapes)]
        s = ''
        for row in rows:
            #s += format % tuple(row) + '\n'
            s += ''.join(row) + '\n'
        return s

    def _mtwords(self):
        applyf = foma_apply_upper_words
        if not self.fsthandle:
            raise ValueError('FST not defined')
        applyerhandle = foma_apply_init(self.fsthandle)
        toksym = '\x07'
        foma_apply_set_space_symbol(c_void_p(applyerhandle), c_char_p(self.encode(toksym)))
        output = applyf(c_void_p(applyerhandle))
        while True:
            if output is None:
                foma_apply_clear(c_void_p(applyerhandle))
                return
            else:
                yield self._fmt(self.decode(output)[:-1].split('\x07'))
            output = applyf(c_void_p(applyerhandle))
