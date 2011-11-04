#lang scheme
(require srfi/1)

(provide multiple?)
(provide sum)
(provide sum-3-5-multiples)
(provide prime?)
(provide list-primes)

; Is x a multiple of y?
(define (multiple? x y) 
  (= 0 (modulo x y)))

; Sum the elements of a numeric list
(define (sum lst)
  (reduce + 0 lst))

; Sum all elements in range [upper, lower) that are multiples of 3 or 5
(define (sum-3-5-multiples upper)
  (sum (filter (lambda (x) 
                 (or (multiple? x 3)
                     (multiple? x 5)))
               (iota upper))))

; Is x prime?
(define (prime? x)
  (let ([possible-factors (iota (truncate (sqrt x)) 2)])
    (not (ormap multiple? 
                (make-list (length possible-factors) x)
                possible-factors))))

; List primes
(define (list-primes count start)
  (filter prime? (iota count start)))
