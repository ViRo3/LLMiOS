import Foundation

public struct ModelConfiguration {
    public let id: String

    public let tokenizerId: String?

    public let overrideTokenizer: String?

    private let preparePrompt: ((String) -> String)?

    public init(
        id: String, tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        preparePrompt: ((String) -> String)? = nil
    ) {
        self.id = id
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.preparePrompt = preparePrompt
    }

    public func prepare(prompt: String) -> String {
        preparePrompt?(prompt) ?? prompt
    }

    public static var registry = [String: ModelConfiguration]()

    public static func register(configurations: [ModelConfiguration]) {
        bootstrap()

        for c in configurations {
            registry[c.id] = c
        }
    }

    public static func configuration(id: String) -> ModelConfiguration {
        bootstrap()

        if let c = registry[id] {
            return c
        } else {
            return ModelConfiguration(id: id)
        }
    }
}

extension ModelConfiguration {
    
    public static let llama3IT = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        overrideTokenizer: "PreTrainedTokenizer") { prompt in
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a friend of the user. Pretend to be a humand and use \
            the entire conversation provided to you along with the last reply\
            and respond to it in the context of the \
            conversation <|eot_id|><|start_header_id|>user<|end_header_id|>
            \(prompt)<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        }
    
    public static let miniCPM = ModelConfiguration(
        id: "mlx-community/MiniCPM-2B-sft-4bit-llama-format-mlx",
        overrideTokenizer: "Pretrained")

    private enum BootstrapState {
        case idle
        case bootstrapping
        case bootstrapped
    }

    static private var bootstrapState = BootstrapState.idle

    static func bootstrap() {
        switch bootstrapState {
        case .idle:
            bootstrapState = .bootstrapping
            register(configurations: [
                llama3IT,
                miniCPM
            ])
            bootstrapState = .bootstrapped

        case .bootstrapping:
            break

        case .bootstrapped:
            break
        }
    }
}
