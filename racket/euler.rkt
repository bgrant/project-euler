#lang racket

; :author: Robert David Grant <robert.david.grant@gmail.com>
;
; :copyright:
;   Copyright 2011 Robert David Grant
;
;   Licensed under the Apache License, Version 2.0 (the "License");
;   you may not use this file except in compliance with the License.
;   You may obtain a copy of the License at
;
;      http://www.apache.org/licenses/LICENSE-2.0
;
;   Unless required by applicable law or agreed to in writing, software
;   distributed under the License is distributed on an "AS IS" BASIS,
;   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;   See the License for the specific language governing permissions and
;   limitations under the License.


(require srfi/1)


;;; Helper functions

(define (multiple? x y) 
  (= 0 (modulo x y)))

(define (divides? x y) 
  (= 0 (modulo y x)))

(define (sum lst)
  (reduce + 0 lst))


;;; Problems

; Sum all elements in range [1, ulimit] that are multiples of 3 or 5
(define (problem1 ulimit)
  (sum (filter (lambda (x) 
                 (or (multiple? x 3)
                     (multiple? x 5)))
               (iota (- ulimit 1) 1))))
