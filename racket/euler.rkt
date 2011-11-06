#lang racket
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