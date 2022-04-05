async def head(cnt, src):
    res = []
    while len(res) < cnt:
        line = await read_line(src)
        if line is None:
            break
        res.append(line)
    return res
