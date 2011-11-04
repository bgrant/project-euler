-- nextFib :: [Integer] -> [Integer]
-- nextFib [] = [1]
-- nextFib [1] = [2,1]
-- nextFib (x2:x1:xs) = (x1 + x2):x2:x1:xs

iterFib :: (Integral a) => [a] -> [a] -> [[a], [a]] 
iterFib
iterFib @allComp(fn2:fn1:computed) @allEnum(e1:enum) = 
    iterFib (fn1 + fn2):allComp (e1 + 1):allEnum

allFibs :: Integer -> [Integer] -> [Integer]
allFibs limit fibs enum = 
    if (head xs) <= limit
    then allFibs limit (nextFib xs)
    else (tail xs)

prob25 = allFibs 
