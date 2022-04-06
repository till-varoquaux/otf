async def head(cnt, src):
    res = []
    while len(res) < cnt:
        line = await src.read_line()
        if line is None:
            break
        res.append(line)
    return res
