eratoSieve :: [Int] -> [Int]
eratoSieve [] = []
eratoSieve (p:xs) = p : eratoSieve [x | x <- xs, x `mod` p > 0]

eulerSieve :: [Int] -> [Int]
eulerSieve [] = []
eulerSieve all@(p:xs) = p : eulerSieve [x | x <- xs, x `notElem` (map (*p) all)]

eratoPrimes lim = eratoSieve [2..lim]
main = print . sum $ eratoPrimes 2000000
