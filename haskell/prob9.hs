prob9 = product . head $ take 1 [[a,b,c] | 
    a <- [1..999], 
    b <- [a..999], 
    c <- [sqrt(a^2 + b^2)],
    a + b + c == 1000]
