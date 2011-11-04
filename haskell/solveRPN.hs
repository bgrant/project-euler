import Data.List

solveRPN :: (Num a) => String -> a
solveRPN expression = head (foldl foldingfunction [] (words expression))
    where foldingFunction stack item = 
