# -*- coding:utf-8 _*-
apcount = {}
word_counts = {}
feedlist = [line for line in open("feedlist.txt")]
for feedurl in feedlist:
    title, wc = get_word_counts(feedurl)
    word_counts[title] = wc
    for word, count in wc.items():
        apcount.setdefault(word, 0)
        if count > 1:
            apcount[word] += 1

word_list = []
for w, bc in apcount.items():
    frac = float(bc) / len(feedlist)
    if 0.1 < frac < 0.5:
        word_list.append(w)

out = open("blogdata.txt", "w")
out.write("Blog")
for word in word_list:
    out.write(f"\t{word}")
out.write("\n")
for blog, wc in word_counts.items():
    out.write(blog)
    for word in word_list:
        if word in wc:
            out.write(f"\t{wc[word]}")
        else:
            out.write("\t0")
    out.write("\n")
