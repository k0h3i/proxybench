"""Display local XML source values as a readable passage, without model labels."""
import html
import re

from proxybench.sources.npx_xml import FORM_NS, VOTE_NS, parse_xml

NAMES = {'issuerName': 'Issuer', 'voteDescription': 'Proposal', 'meetingDate': 'Meeting date',
         'cusip': 'CUSIP', 'isin': 'ISIN', 'voteSource': 'Proposal source',
         'sharesVoted': 'Shares voted', 'sharesOnLoan': 'Shares on loan',
         'howVoted': 'Vote cast', 'managementRecommendation': 'Management alignment',
         'voteSeries': 'Reporting series ID', 'idOfSeries': 'Series ID',
         'nameOfSeries': 'Fund name', 'leiOfSeries': 'Legal entity identifier',
         'voteRecord': 'Vote component', 'vote': 'Disclosed votes', 'seriesReports': 'Reporting fund'}


def heading(name):
    return NAMES.get(name, re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name).capitalize())


def descendants(node, name):
    found = [node] if node.name == name else []
    for child in node.children:
        found.extend(descendants(child, name))
    return found


def display(node):
    title = html.escape(heading(node.name))
    if not node.children:
        return '<div class="fact"><dt>' + title + '</dt><dd>' + html.escape(node.text) + '</dd></div>'
    text = '<p>' + html.escape(node.text) + '</p>' if node.text.strip() else ''
    return '<section class="group"><h3>' + title + '</h3>' + text + '<dl>' + ''.join(display(c) for c in node.children) + '</dl></section>'


def passage(primary, votes, *, index):
    form = parse_xml(primary, namespace=FORM_NS, root_name='edgarSubmission')
    table = parse_xml(votes, namespace=VOTE_NS, root_name='proxyVoteTable')
    if type(index) is not int or not 0 <= index < len(table.children):
        raise ValueError('Invalid source target')
    target = table.children[index]
    if target.name != 'proxyTable':
        raise ValueError('Expected a vote entry')
    issuer = target.one('issuerName', optional=True)
    description = target.one('voteDescription', optional=True)
    series_ids = {n.text for n in target.all('voteSeries')}
    series = [node for node in descendants(form, 'seriesReports')
              if any(child.text in series_ids for child in node.all('idOfSeries'))]
    context = ''.join(display(node) for node in series)
    if not context:
        context = '<p>No matching series entry appears in the primary attachment.</p>'
    title = html.escape(issuer.text) if issuer else 'Selected voting record'
    proposal = html.escape(description.text) if description else 'No proposal description appears in this entry.'
    other = ''.join(display(node) for node in target.children if node.name not in {'issuerName', 'voteDescription'})
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;padding:24px;color:#192b37;background:#fff;font:16px/1.55 system-ui,sans-serif}
h1{font-size:24px;margin:4px 0 16px}h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:12px 0 8px}
.caption{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#586b78}
.proposal{background:#fff5ce;border-left:4px solid #d09b22;padding:18px 20px;margin:14px 0 24px;white-space:pre-wrap;font-size:18px}
.columns{display:grid;grid-template-columns:3fr 2fr;gap:24px}.context{background:#f1f5f7;padding:18px;border-radius:8px}
dl{margin:0}.fact{display:grid;grid-template-columns:minmax(120px,1fr) 2fr;gap:12px;padding:8px 0;border-bottom:1px solid #e4e9ec}
dt{color:#506471;font-size:14px}dd{margin:0;overflow-wrap:anywhere;white-space:pre-wrap}.group{margin:10px 0 18px}
details{margin-top:24px;border-top:1px solid #d6dfe4;padding-top:16px}summary{cursor:pointer;color:#164c68}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.note{font-size:13px;color:#586b78}
@media(max-width:700px){body{padding:16px}.columns{grid-template-columns:1fr}}
</style><body><div class="caption">Original filing · selected vote</div><h1>''' + title + '''</h1>
<h2>Proposal passage</h2><div class="proposal">''' + proposal + '''</div>
<div class="columns"><section><h2>Disclosed voting details</h2><dl>''' + other + '''</dl></section>
<aside class="context"><h2>Reporting fund context</h2>''' + context + '''</aside></div>
<p class="note">This view displays the source values with readable headings. It contains no model answers. The alignment value is shown as filed.</p>
<details><summary>Full primary attachment as readable fields</summary>''' + display(form) + '''</details>
<details><summary>Original XML</summary><pre>''' + html.escape(primary.decode()) + '</pre><pre>' + html.escape(votes[target.start:target.end].decode()) + '</pre></details></body></html>'
