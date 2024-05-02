import Foundation

public enum StringOrNumber: Codable, Equatable {
    case string(String)
    case float(Float)

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        }
    }
}

public enum ModelType: String, Codable {
    case llama
    
    public func createModel(configuration: URL) throws -> LLMModel {
        switch self {
        case .llama:
            let configuration = try JSONDecoder().decode(
                LlamaConfiguration.self, from: Data(contentsOf: configuration))
            return LlamaModel(configuration)
        }
    }
}

public struct BaseConfiguration: Codable {
    public let modelType: ModelType

    public struct Quantization: Codable {
        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        let groupSize: Int
        let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
        }
    }

    public var quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
    }
}
