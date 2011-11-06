#lang racket
(require srfi/1)


;;; Helper functions

(define (multiple? x y) 
  (= 0 (modulo x y)))

(define (divides? x y) 
  (= 0 (modulo y x)))

(define (sum lst)
  (reduce + 0 lst))

; Check primality by trial division
(define (prime? x)
  (let ([possible-factors (iota (truncate (sqrt x)) 2)])
    (not (ormap multiple? 
                (make-list (length possible-factors) x)
                possible-factors))))

; List all primes up to end
(define (list-primes end)
  (filter prime? (iota end)))

(define (prime-factors n)
  (filter (lambda (x)
            (and (divides? x n)
                 (prime? x)))))

;;; Problems

; Sum all elements in range [1, ulimit] that are multiples of 3 or 5
(define (problem1 ulimit)
  (sum (filter (lambda (x) 
                 (or (multiple? x 3)
                     (multiple? x 5)))
               (iota ulimit))))

(define (problem3 n)
  (tail (prime-factors n)))