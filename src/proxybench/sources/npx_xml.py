"""Bounded, non-resolving XML reads with original byte locations."""

from dataclasses import dataclass, field
from xml.parsers import expat
import hashlib

VOTE_NS = "http://www.sec.gov/edgar/document/npxproxy/informationtable"
FORM_NS = "http://www.sec.gov/edgar/npx"
COMMON_NS = "http://www.sec.gov/edgar/common"


class XMLRejected(ValueError):
    pass


@dataclass
class Node:
    name: str
    namespace: str
    start: int
    end: int = 0
    text: str = ""
    children: list = field(default_factory=list)

    def all(self, name):
        return [n for n in self.children if n.name == name]

    def one(self, name, optional=False):
        found = self.all(name)
        if optional and not found:
            return None
        if len(found) != 1:
            raise XMLRejected(f"Expected one {name}, found {len(found)} at byte {self.start}")
        return found[0]

    def at(self, path):
        node = self
        for name in path.split("/"):
            node = node.one(name)
        return node


def parse_xml(raw, *, namespace, root_name, max_bytes=16 * 1024 * 1024,
              max_depth=32, max_nodes=200000, max_records=10000):
    if not isinstance(raw, bytes) or not raw or len(raw) > max_bytes:
        raise XMLRejected("Empty XML or file size limit exceeded")
    # Byte evidence uses UTF-8. ASCII declarations are also valid UTF-8 bytes.
    try:
        raw.decode("utf-8")
    except UnicodeError as error:
        raise XMLRejected("Unsupported XML encoding") from error
    parser = expat.ParserCreate(namespace_separator="}")
    stack, roots = [], []
    counts = {"nodes": 0, "records": 0, "text": 0}

    def reject(*args):
        raise XMLRejected("DTD, entity declarations, and external resolution are prohibited")

    def start(name, attrs):
        ns, _, local = name.rpartition("}")
        allowed = {namespace} | ({COMMON_NS} if namespace == FORM_NS else set())
        if ns not in allowed or attrs:
            raise XMLRejected("Unknown namespace or unsupported attributes")
        counts["nodes"] += 1
        counts["records"] += local == "proxyTable"
        if len(stack) >= max_depth or counts["nodes"] > max_nodes or counts["records"] > max_records:
            raise XMLRejected("XML depth, node, or record limit exceeded")
        node = Node(local, ns, parser.CurrentByteIndex)
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)

    def end(name):
        node = stack.pop()
        pos = parser.CurrentByteIndex
        node.end = raw.index(b">", pos) + 1 if raw[pos:pos + 2] == b"</" else pos

    def data(text):
        counts["text"] += len(text.encode("utf-8"))
        if counts["text"] > max_bytes:
            raise XMLRejected("Expanded text limit exceeded")
        if stack:
            stack[-1].text += text

    parser.StartElementHandler, parser.EndElementHandler = start, end
    parser.CharacterDataHandler = data
    parser.StartDoctypeDeclHandler = reject
    parser.EntityDeclHandler = reject
    parser.ExternalEntityRefHandler = reject
    parser.XmlDeclHandler = lambda version, encoding, standalone: (
        reject() if encoding and encoding.lower() not in {"utf-8", "us-ascii", "ascii"} else None)
    try:
        parser.Parse(raw, True)
    except (expat.ExpatError, UnicodeError, ValueError) as error:
        raise XMLRejected(str(error)) from error
    if len(roots) != 1 or roots[0].name != root_name or roots[0].namespace != namespace:
        raise XMLRejected("Unexpected XML root")
    return roots[0]


def pointer(node, raw, document_id):
    return {"document_id": document_id, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "start_byte": node.start, "end_byte": node.end}
