collatzSeq n
    | (even n) = n : collatzSeq (n `quot` 2)
    | (odd n)  = n : collatzSeq (3*n + 1)

collatzLen n = (length (takeWhile (>1) (collatzSeq n))) + 1

collatzPairs = [(n,collatzLen n) | n <- [1..1000000]]

-- prob14 = [n | n <- collatzPairs, snd n == maximum (map snd collatzPairs)]

maxPair :: [(Integer, Int)] -> (Integer, Int)
maxPair xs = foldl1 (\x y -> if (snd x > snd y) then x else y) xs

main = do print (maxPair collatzPairs)
